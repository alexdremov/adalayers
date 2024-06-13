from dataclasses import dataclass, field
from omegaconf import MISSING

from typing import Optional


@dataclass
class OptimizationConfig:
    optimizer: str = "adam"
    optim_kwargs: dict = field(default_factory=lambda: dict(lr=1e-4, weight_decay=0.0))
    batch_size: int = 2
    batch_size_eval: int = 4
    num_workers: int = 0
    max_epochs: int = 32
    best_metric: str = "f1"
    lr_patience: int = 5
    mode: str = "default"
    early_stop_patience: int = 10
    precision: Optional[str] = None
    min_delta: float = 0.0


@dataclass
class ModelConfig:
    name: str = MISSING
    kwargs: dict = field(default_factory=dict)
    restore_artifact: Optional[str] = None


@dataclass
class WandbConfig:
    project: str = "adalayers"
    name: str = MISSING
    notes: str = ""


@dataclass
class DatasetConfig:
    name: str = MISSING
    num_classes: int = 2
    save_eval_data: bool = False


@dataclass
class Experiment:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    tokenizer_pretrained: dict = field(default_factory=dict)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    seed: int = 42
