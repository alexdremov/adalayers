from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class OptimizationConfig:
    optimizer: str = 'adam'
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 2
    num_workers: int = 0
    max_epochs: int = 32


@dataclass
class ModelConfig:
    name: str = MISSING
    kwargs: dict = field(default_factory=dict)


@dataclass
class WandbConfig:
    project: str = "adalayers"
    name: str = MISSING
    notes: str = ""


@dataclass
class DatasetConfig:
    name: str = MISSING
    num_classes: int = 2


@dataclass
class Experiment:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    tokenizer_pretrained: dict = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
