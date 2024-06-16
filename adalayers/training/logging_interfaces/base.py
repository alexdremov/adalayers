from abc import ABC, abstractmethod

from pytorch_lightning.loggers import Logger

class BaseLogger(Logger, ABC):
    @abstractmethod
    def log_code(self, dir):
        ...

    @abstractmethod
    def log_configs(self, dir):
        ...

    @abstractmethod
    def finalize(self, status):
        ...

    @abstractmethod
    def get_artifact(self, name):
        ...

    @abstractmethod
    def log_model_checkpoint(self, name, metadata, description, dir):
        ...

    @abstractmethod
    def log_artifact(self, name, metadata, description, object):
        ...

    @abstractmethod
    def set_summary(self, summary):
        ...
