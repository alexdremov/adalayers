from .clearml import ClearmlLogger
from .wandb import WandbLogger
from .none import NoneLogger


_logger: NoneLogger | ClearmlLogger | WandbLogger | None = None


def register_logger(type, **kwargs):
    global _logger
    assert _logger is None

    cls = NoneLogger
    match type:
        case "none":
            cls = NoneLogger
        case "wandb":
            cls = WandbLogger
        case "clearml":
            cls = ClearmlLogger
        case _:
            raise RuntimeError("Unknown logger")
    _logger = cls(**kwargs)
    return _logger


def get_logger() -> NoneLogger | ClearmlLogger | WandbLogger:
    assert _logger is not None, "Must setup logging"
    return _logger
