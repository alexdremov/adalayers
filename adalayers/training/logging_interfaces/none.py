from .base import BaseLogger

from pytorch_lightning.loggers.logger import DummyLogger

class NoneLogger(BaseLogger, DummyLogger):
    def __init__(
        self,
        **kwargs
    ):
        DummyLogger.__init__(self)

    def log_code(self, dir):
        ...

    def finalize(self, status):
        ...

    def get_artifact(self, name):
        return None

    def log_model_checkpoint(self, name, metadata, description, dir):
        ...

    def log_artifact(self, name, metadata, description, object):
        ...

    def set_summary(self, summary):
        ...

    def log_model_object(self, name, metadata, config, description, object):
        ...
