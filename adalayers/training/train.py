import logging
import functools

import lightning as pl
from lightning.pytorch.loggers import WandbLogger

import transformers
import torchmetrics

import torch
import torch.nn as nn
import torch.utils.data
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from transformers.modeling_outputs import SequenceClassifierOutput

from adalayers.training.config import Experiment

logger = logging.getLogger(__name__)


def collate_fn(batch, collator):
    return collator(batch)


class LightningModel(pl.LightningModule):
    def __init__(self, model, tokenizer, dataset, experiment: Experiment):
        super().__init__()

        self.save_hyperparameters(experiment)
        self.model = model
        self.experiment = experiment

        self.batch_size = experiment.optimization.batch_size
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

        self.acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=experiment.dataset.num_classes
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
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

        self.log('val/loss', output.loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val/acc', self.acc, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        self.log('val/f1', self.f1, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        return output.loss

    def on_validation_epoch_end(self):
        logger.info(
            dict(
                acc=self.acc.compute(),
                f1=self.f1.compute()
            )
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset['train'],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=functools.partial(collate_fn, collator=self.collator)
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset['val'],
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=functools.partial(collate_fn, collator=self.collator)
        )


def train(
        experiment,
        model,
        tokenizer,
        dataset,
        root_dir
):
    logger.info(model)
    logger.info(tokenizer)
    logger.info(dataset)

    wandb_logger = WandbLogger(
        log_model="all",
        save_dir=root_dir,
        project=experiment.wandb.project,
    )
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        accelerator='gpu',
        logger=wandb_logger,
        max_epochs=experiment.optimization.max_epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
        default_root_dir=root_dir
    )

    pl_model = LightningModel(model, tokenizer, dataset, experiment)
    trainer.fit(
        pl_model
    )
