from typing import Dict

import tempfile

import os
import torch
from torch import Tensor
from torch.nn import Module

from .base import BaseLogger

import os
from clearml import Logger, Task, OutputModel

class ClearmlLogger(BaseLogger):
    def __init__(
        self,
        log_model,
        save_dir,
        project,
        config,
        name,
        notes,
    ):
        self.task: Task = Task.init(
            project_name=project,
            task_name=name,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
        )
        self._save_dir = save_dir
        self.logger: Logger = self.task.get_logger()
        self.task.set_parameters_as_dict(
            dictionary=config
        )
        self.task.set_comment(notes)

    def log_code(self, dir):
        return self.task.upload_artifact(
            name='code',
            artifact_object=os.path.join(dir)
        )


    def finalize(self, status):
        self.log_model_checkpoint(
            name="result_dir",
            metadata=dict(),
            description="results directory",
            dir=self._save_dir,
        )
        self.task.add_tags(status)
        self.task.publish_on_completion()
        self.save()

    def get_artifact(self, name):
        task_id, artifact = name.split('/')
        dep_task = Task.get_task(task_id=task_id)
        return dep_task.artifacts[artifact].get_local_copy()

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        for name, value in metrics.items():
            if name.count('/') > 0:
                title, series = name.split('/', 1)
            else:
                series, title = "general", name
            self.logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=step or 0,
            )

    def log_model_checkpoint(self, name, metadata, description, dir):
        assert name not in self.task.artifacts, f"Model with {name = } already exists"
        output_model = OutputModel(
            task=self.task,
            framework="PyTorch",
            config_dict=metadata,
            name=name,
            comment=description,
        )
        output_model.update_weights_package(
            weights_path=dir,
            auto_delete_file=False,
        )

    def log_artifact(self, name, metadata, description, object):
        assert name not in self.task.artifacts, f"Model with {name = } already exists"
        return self.task.upload_artifact(
            name=name,
            artifact_object=object,
            metadata=metadata | dict(description=description),
        )

    def log_graph(self, model: Module, input_array: Tensor | None = None) -> None:
        output_model = OutputModel(
            task=self.task,
            framework="PyTorch"
        )
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as file:
            torch.save(model, file)
        output_model.update_weights(
            weights_filename=file.name
        )
        output_model.update_design(
            config_text=str(model)
        )

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        self.task.connect(params, name='hyperparameters')
        self.task.connect(
            dict(
                args=args,
                kwargs=kwargs
            ),
            name='args'
        )

    @property
    def name(self):
        return self.task.name

    @property
    def version(self):
        return 0

    def save(self) -> None:
        super().save()
        self.task.flush(wait_for_uploads=False)

    @property
    def save_dir(self):
        return self._save_dir

    def set_summary(self, summary):
        self.task.connect(summary, name='summary')
