import logging
import os
import sys

import lightning as pl

import transformers
import torchmetrics

import torch
import torch.utils.data

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers.modeling_outputs import SequenceClassifierOutput

from adalayers.models.ada_layers_classifier import AdaLayersForSequenceClassification
from adalayers.training.config import Experiment

logger = logging.getLogger(__name__)


class LightningModel(pl.LightningModule):
    def __init__(self, model, tokenizer, dataset, experiment: Experiment):
        super().__init__()

        self.save_hyperparameters(experiment)
        self.model = model
        self.experiment = experiment

        self.print_distribution = True

        self.batch_size = experiment.optimization.batch_size
        self.batch_size_eval = experiment.optimization.batch_size_eval
        self.num_workers = experiment.optimization.num_workers

        self.collator = transformers.DataCollatorWithPadding(tokenizer)
        self.dataset = dataset.select_columns(['input_ids', 'label', 'attention_mask'])

        self.f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average='macro'
        )
        self.valid_f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average='macro'
        )
        self.test_f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average='macro'
        )

        self.acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=experiment.dataset.num_classes
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=experiment.dataset.num_classes
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=experiment.dataset.num_classes
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        match self.experiment.optimization.optimizer:
            case 'adam':
                optimizer_cls = torch.optim.Adam
            case 'adamw':
                optimizer_cls = torch.optim.AdamW
            case _:
                assert False, 'unknown optimizer type'

        optimizer = optimizer_cls(
            self.parameters(),
            **self.experiment.optimization.optim_kwargs
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        output: SequenceClassifierOutput = self(
            **batch,
            output_hidden_states=True
        )

        self.acc.update(output.logits, batch['labels'])
        self.f1.update(output.logits, batch['labels'])

        self.log('train/loss', output.loss, prog_bar=True, on_step=True)
        self.log('train/acc', self.acc, prog_bar=True, on_epoch=True, on_step=True)
        self.log('train/f1', self.f1, prog_bar=True, on_epoch=True, on_step=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(
            **batch
        )

        self.valid_acc.update(output.logits, batch['labels'])
        self.valid_f1.update(output.logits, batch['labels'])

        name = 'val'
        self.log(f'{name}/loss', output.loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{name}/acc', self.acc, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        self.log(f'{name}/f1', self.f1, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        return output.loss

    def test_step(self, batch, batch_idx):
        output = self(
            **batch
        )

        self.test_acc.update(output.logits, batch['labels'])
        self.test_f1.update(output.logits, batch['labels'])

        name = 'test'
        self.log(f'{name}/loss', output.loss, on_epoch=True, sync_dist=True)
        self.log(f'{name}/acc', self.acc, on_epoch=True, sync_dist=True)
        self.log(f'{name}/f1', self.f1, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        output = self(
            **batch
        )
        return output.logits

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        if isinstance(self.model, AdaLayersForSequenceClassification) and self.print_distribution:
            distribution = self.model.distribution_normalized
            logger.info("Adaptive layers distribution:")
            logger.info(distribution.detach().cpu().view(-1))

    def on_validation_epoch_end(self):
        metrics = dict(
            acc=self.acc.compute(),
            f1=self.f1.compute()
        )
        if self.trainer.is_global_zero:
            logger.info(
                metrics
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=self.batch_size_eval,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=self.collator
        )


def train(
        experiment: Experiment,
        model,
        tokenizer,
        dataset,
        root_dir,
        wandb_logger
):
    logger.info(model)
    logger.info(tokenizer)
    logger.info(dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(root_dir, 'lightning_checkpoints'),
        filename='{epoch}-{val/loss:.2f}-{val/' + experiment.optimization.best_metric + ':.2f}',
        monitor=f'val/{experiment.optimization.best_metric}',
        mode='max',
        save_weights_only=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        logger=wandb_logger,
        max_epochs=experiment.optimization.max_epochs,
        default_root_dir=root_dir,
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback
        ]
    )

    pl_model = LightningModel(model, tokenizer, dataset, experiment)

    logger.info("Start train")
    trainer.fit(
        pl_model
    )
    logger.info("End train")

    last_path = os.path.join(root_dir, 'lightning_checkpoints/last.ckpt')
    trainer.save_checkpoint(last_path)

    if not trainer.is_global_zero:
        return None
    return (
        checkpoint_callback.best_model_path,
        last_path,
    )
