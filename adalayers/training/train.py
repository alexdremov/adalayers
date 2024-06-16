import logging
import os
import datetime

import lightning as pl

from omegaconf import OmegaConf
import transformers
import torchmetrics

import torch
import torch.utils.data

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers.modeling_outputs import SequenceClassifierOutput

from adalayers.training.config import Experiment
from adalayers.training.logging_interfaces.factory import get_logger

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
        self.mode = experiment.optimization.mode

        match self.mode:
            case "default":
                collator = transformers.DataCollatorWithPadding(
                    tokenizer, padding="longest", max_length=tokenizer.model_max_length
                )
            case "conll_ner":
                collator = transformers.DataCollatorForTokenClassification(
                    tokenizer, padding="longest", max_length=tokenizer.model_max_length
                )

        self.collator = collator

        self.columns_to_use = ["input_ids", "attention_mask"]
        if "label" in dataset.column_names["train"]:
            self.columns_to_use.append("label")
        if "labels" in dataset.column_names["train"]:
            self.columns_to_use.append("labels")
        self.dataset = dataset.select_columns(self.columns_to_use)

        self.f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="macro",
        )
        self.valid_f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="macro",
        )
        self.test_f1 = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="macro",
        )

        self.f1_micro = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="micro",
        )
        self.valid_f1_micro = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="micro",
        )
        self.test_f1_micro = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=experiment.dataset.num_classes,
            average="micro",
        )

        self.acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=experiment.dataset.num_classes
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=experiment.dataset.num_classes
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=experiment.dataset.num_classes
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        match self.experiment.optimization.optimizer:
            case "adam":
                optimizer_cls = torch.optim.Adam
            case "adamw":
                optimizer_cls = torch.optim.AdamW
            case _:
                assert False, "unknown optimizer type"

        optimizer_kwargs = self.experiment.optimization.optim_kwargs
        if OmegaConf.is_config(optimizer_kwargs):
            optimizer_kwargs: dict = OmegaConf.to_container(
                optimizer_kwargs, resolve=True
            )

        optimizer = optimizer_cls(
            [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if p.requires_grad and not n.endswith("distribution")
                    ],
                    **optimizer_kwargs,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if p.requires_grad and n.endswith("distribution")
                    ],
                    **(
                        optimizer_kwargs
                        | {
                            "weight_decay": 0.0,
                        }
                    ),
                },
            ]
        )
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    verbose=True,
                    factor=0.5,
                    min_lr=1e-6,
                    patience=self.experiment.optimization.lr_patience,
                ),
                "monitor": f"val/{self.experiment.optimization.best_metric}",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def training_step(self, batch, batch_idx):
        output: SequenceClassifierOutput = self(**batch)

        name = "train"
        self.update_metrics(output.logits, batch["labels"], step=name)
        self.log(f"{name}/loss", output.loss, prog_bar=True, on_step=True)
        self.log(f"{name}/acc", self.acc, prog_bar=True, on_epoch=True)
        self.log(f"{name}/f1", self.f1, prog_bar=True, on_epoch=True)
        self.log(f"{name}/f1_micro", self.f1_micro, prog_bar=True, on_epoch=True)

        if (
            hasattr(self.model, "distribution_normalized")
            and self.print_distribution
            and self.trainer.is_global_zero
        ):
            with torch.no_grad():
                distribution = self.model.distribution_normalized
                run_logger = get_logger()
                run_logger.log_metrics(
                    {
                        f"distribution/layer_{i}": value
                        for i, value in enumerate(
                            distribution.detach().cpu().view(-1).numpy().tolist()
                        )
                    },
                    step=batch_idx
                )

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        name = "val"
        self.update_metrics(output.logits, batch["labels"], step=name)
        self.log(
            f"{name}/loss", output.loss, prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{name}/acc", self.valid_acc, prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{name}/f1", self.valid_f1, prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{name}/f1_micro",
            self.valid_f1_micro,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return output.loss

    def test_step(self, batch, batch_idx):
        output = self(**batch)

        name = "test"
        self.update_metrics(output.logits, batch["labels"], step=name)
        self.log(f"{name}/loss", output.loss, on_epoch=True, sync_dist=True)
        self.log(f"{name}/acc", self.test_acc, on_epoch=True, sync_dist=True)
        self.log(f"{name}/f1", self.test_f1, on_epoch=True, sync_dist=True)
        self.log(f"{name}/f1_micro", self.test_f1_micro, on_epoch=True, sync_dist=True)

    def update_metrics(self, logits, labels, step="train"):
        acc, f1, f1_micro = (self.acc, self.f1, self.f1_micro)
        if step == "val":
            acc, f1, f1_micro = (self.valid_acc, self.valid_f1, self.valid_f1_micro)
        elif step == "test":
            acc, f1, f1_micro = (self.test_acc, self.test_f1, self.test_f1_micro)

        if self.mode == "default":
            acc.update(logits, labels)
            f1.update(logits, labels)
            f1_micro.update(logits, labels)
            return

        logits, labels = logits.view(-1, logits.size(-1)), labels.view(-1)
        mask_valid = labels != -100
        logits = logits[mask_valid]
        labels = labels[mask_valid]

        acc.update(logits, labels)
        f1.update(logits, labels)
        f1_micro.update(logits, labels)

    def predict_step(self, batch, batch_idx):
        output = self(**batch)
        return output.logits

    def on_train_epoch_end(self):
        metrics = dict(
            acc=self.acc.compute(),
            f1=self.f1.compute(),
            f1_micro=self.f1_micro.compute(),
        )
        if not self.trainer.is_global_zero:
            return

        logger.info("Train epoch results")
        logger.info(metrics)
        if hasattr(self.model, "distribution_normalized") and self.print_distribution:
            distribution = self.model.distribution_normalized
            logger.info("Adaptive layers distribution:")
            logger.info(distribution.detach().cpu().view(-1))

    def on_validation_epoch_end(self):
        metrics = dict(
            acc=self.valid_acc.compute(),
            f1=self.valid_f1.compute(),
            f1_micro=self.valid_f1_micro.compute(),
        )
        if not self.trainer.is_global_zero:
            return
        logger.info("Validation epoch results")
        logger.info(metrics)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dataset["val"],
            batch_size=self.batch_size_eval,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )


def train(experiment: Experiment, model, tokenizer, dataset, root_dir):
    logger.info(model)
    logger.info(tokenizer)
    logger.info(dataset)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "lightning_checkpoints"),
        filename=("{epoch}-{val/" + experiment.optimization.best_metric + ":.4f}"),
        monitor=f"val/{experiment.optimization.best_metric}",
        mode="max",
        save_weights_only=True,
        verbose=True,
        save_last=False,
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stop_callback = EarlyStopping(
        monitor=f"val/{experiment.optimization.best_metric}",
        min_delta=experiment.optimization.min_delta,
        patience=experiment.optimization.early_stop_patience,
        verbose=True,
        mode="max",
    )

    strategy = 'auto'
    precision= None
    if torch.cuda.is_available():
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=datetime.timedelta(seconds=180000),
        )
        precision = experiment.optimization.precision


    run_logger = get_logger()
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=run_logger,
        max_epochs=experiment.optimization.max_epochs,
        default_root_dir=root_dir,
        log_every_n_steps=2,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        strategy=strategy,
        precision=precision,
    )

    pl_model = LightningModel(model, tokenizer, dataset, experiment)

    logger.info("Start train")
    trainer.fit(pl_model)
    logger.info("End train")

    last_path = os.path.join(root_dir, "lightning_checkpoints/last.ckpt")
    trainer.save_checkpoint(last_path)

    if not trainer.is_global_zero:
        return None
    return (
        checkpoint_callback.best_model_path,
        last_path,
    )
