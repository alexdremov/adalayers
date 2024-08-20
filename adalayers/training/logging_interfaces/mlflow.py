from typing import Dict

import tempfile
import atexit

import os
import torch
from torch import Tensor
from torch.nn import Module

import mlflow

from .base import BaseLogger

import os

class MlflowLogger(BaseLogger):
    def __init__(
        self,
        log_model,
        save_dir,
        project,
        config,
        name,
        notes,
    ):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=name, description=notes)
        self._name = name
        self._save_dir = save_dir


    def log_code(self, dir):
        return mlflow.log_artifacts(dir, 'code')

    def log_configs(self, dir):
        return mlflow.log_artifacts(dir, 'configs')

    def finalize(self, status):
        self.save()
        mlflow.end_run(status)

    def get_artifact(self, name):
        return mlflow.pyfunc.load_model(name)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step)

    def log_model_checkpoint(self, name, metadata, description, dir):
        mlflow.log_artifacts(dir, name)

    def log_artifact(self, name, metadata, description, object):
        with tempfile.NamedTemporaryFile(mode='wb') as file:
            torch.save(object, file)
            mlflow.log_artifact(file.name, name)

    def log_graph(self, model: Module, input_array: Tensor | None = None) -> None:
        mlflow.pytorch.log_model(
            model,
            input_example=input_array
        )

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        mlflow.log_params(
            dict(
                hyperparameters=params,
                args=args,
                kwargs=kwargs,
            )
        )

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return 0

    @property
    def save_dir(self):
        return self._save_dir

    def set_summary(self, summary):
        mlflow.log_dict(summary, "summary")
