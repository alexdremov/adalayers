from argparse import Namespace
from typing import Any, Dict
import tempfile
import pickle

from .base import BaseLogger

import wandb
from wandb.wandb_run import Run
from lightning.pytorch.loggers import WandbLogger as PlWandbLogger


class WandbLogger(BaseLogger):
    experiment: Run

    def __init__(
        self,
        log_model,
        save_dir,
        project,
        config,
        name,
        notes,
    ):
        self.logger = PlWandbLogger(
            log_model=log_model,
            save_dir=save_dir,
            project=project,
            config=config,
            name=name,
            notes=notes,
        )
        self.experiment = self.logger.experiment
        for step in ["train", "val", "test"]:
            self.experiment.define_metric(
                f"{step}/loss_epoch", goal="minimize", summary="min,last"
            )
            self.experiment.define_metric(
                f"{step}/acc_epoch", goal="maximize", summary="max,last"
            )
            self.experiment.define_metric(
                f"{step}/f1_epoch", goal="maximize", summary="max,last"
            )
            self.experiment.define_metric(
                f"{step}/loss", goal="minimize", summary="min,last"
            )
            self.experiment.define_metric(
                f"{step}/acc", goal="maximize", summary="max,last"
            )
            self.experiment.define_metric(
                f"{step}/f1", goal="maximize", summary="max,last"
            )
            self.experiment.define_metric(
                f"{step}/f1_micro", goal="maximize", summary="max,last"
            )

    def log_code(self, dir):
        self.experiment.log_code(dir)

    def log_configs(self, dir):
        artifact = wandb.Artifact(
            name='configs', type="config",
        )
        artifact.add_dir(local_path=dir)
        self.experiment.log_artifact(artifact)

    def finalize(self, status):
        try:
            self.logger.finalize(status=status)
        except:
            ...

    def get_artifact(self, name):
        result = self.experiment.use_artifact(name)
        if result is not None:
            return result.download()
        return None

    def log_model_checkpoint(self, name, metadata, description, dir):
        artifact = wandb.Artifact(
            name=name, type="model", metadata=metadata, description=description
        )
        artifact.add_dir(local_path=dir)
        self.experiment.log_artifact(artifact)

    def log_artifact(self, name, metadata, description, object):
        artifact = wandb.Artifact(
            name=name, type="custom", metadata=metadata, description=description
        )
        with tempfile.NamedTemporaryFile('wb') as file:
            pickle.dump(object, file)
            artifact.add_file(file.name)

    def set_summary(self, summary):
        self.experiment.summary.update(summary)

    def log_hyperparams(self, params: Dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        return self.logger.log_hyperparams(params, *args, **kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        return self.logger.log_metrics(metrics, step)

    @property
    def name(self):
        return self.logger.name

    @property
    def version(self):
        return self.logger.version
