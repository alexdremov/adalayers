import os

import hydra
import hydra.core.hydra_config

import logging
import torch
import random

from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything as pl_seed_everything

import numpy as np

from omegaconf import OmegaConf, DictConfig

from adalayers.training.config import Experiment
from adalayers.models import build_model, build_tokenizer
from adalayers.datasets import build_dataset
from adalayers.training.evaluate import eval_and_save
from adalayers.training.train import train

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """Seed everything to make computations reproducible.

    Args:
        seed (int): Seed value to be used
    """
    pl_seed_everything(seed=seed, workers=True)

    # Python's random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # CuDNN behaviors
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally, you might also want to seed the hash function environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


def process(experiment: Experiment, res_dir: str):
    experiment_resolved = OmegaConf.to_container(
        experiment, resolve=True, throw_on_missing=True
    )
    logger.info(experiment_resolved)
    wandb_logger = WandbLogger(
        log_model=True,
        save_dir=res_dir,
        project="adalayers",
        config=experiment_resolved,
        name=experiment.wandb.name,
        notes=experiment.wandb.notes,
    )
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./..")
    logger.info(f"Code: {code_dir}")
    wandb_logger.experiment.log_code(code_dir)

    model = build_model(experiment, wandb_logger.experiment)
    tokenizer = build_tokenizer(experiment)
    dataset = build_dataset(experiment.dataset.name, tokenizer)

    for step in ["train", "val", "test"]:
        wandb_logger.experiment.define_metric(
            f"{step}/loss_epoch", goal="minimize", summary="min,last"
        )
        wandb_logger.experiment.define_metric(
            f"{step}/acc_epoch", goal="maximize", summary="max,last"
        )
        wandb_logger.experiment.define_metric(
            f"{step}/f1_epoch", goal="maximize", summary="max,last"
        )
        wandb_logger.experiment.define_metric(
            f"{step}/loss", goal="minimize", summary="min,last"
        )
        wandb_logger.experiment.define_metric(
            f"{step}/acc", goal="maximize", summary="max,last"
        )
        wandb_logger.experiment.define_metric(
            f"{step}/f1", goal="maximize", summary="max,last"
        )

    train_res = train(experiment, model, tokenizer, dataset, res_dir, wandb_logger)
    if train_res is not None:  # zero rank
        best_model_path, last_model_path = train_res
        eval_and_save(
            experiment=experiment,
            best_model_path=best_model_path,
            last_model_path=last_model_path,
            dataset=dataset,
            model=model,
            res_dir=res_dir,
            tokenizer=tokenizer,
            wandb_logger=wandb_logger,
        )
    torch.distributed.barrier()
    wandb_logger.finalize(status="success")


OmegaConf.register_new_resolver("cat", lambda *x: " ".join(map(str, x)))


@hydra.main(config_path="../../configs", version_base=None)
def main(cfg: DictConfig):
    experiment: Experiment = OmegaConf.merge(OmegaConf.structured(Experiment), cfg)
    seed_everything(experiment.seed)
    logger.info(experiment.wandb.name)

    res_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"{res_dir = }")
    process(experiment, res_dir)


if __name__ == "__main__":
    main()
