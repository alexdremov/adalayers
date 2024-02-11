import os

import hydra
import hydra.core.hydra_config

import logging
import torch
import random

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything as pl_seed_everything

import numpy as np

from omegaconf import OmegaConf, DictConfig
from wandb.apis.public import Run

from adalayers.training.config import Experiment
from adalayers.models import build_model, build_tokenizer
from adalayers.datasets import build_dataset
from adalayers.training.evaluate import evaluate
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


def dump_wandb_summary_metrics(wandb: Run, results, name, model):
    for k, v in results.items():
        wandb.summary[f"{name}_{model}_{k}"] = v


def process(experiment: Experiment, res_dir: str):
    model = build_model(experiment)

    tokenizer = build_tokenizer(experiment)
    dataset = build_dataset(experiment.dataset.name, tokenizer)

    experiment_resolved = OmegaConf.to_container(
        experiment, resolve=True, throw_on_missing=True
    )
    wandb_logger = WandbLogger(
        log_model="all",
        save_dir=res_dir,
        project="adalayers",
        config=experiment_resolved,
        name=experiment.wandb.name,
        notes=experiment.wandb.notes,
        group=experiment.wandb.name,
    )
    wandb_logger.watch(model)
    for step in ['train', 'val']:
        wandb_logger.experiment.define_metric(f"{step}/loss_epoch", goal="minimize", summary="min,last")
        wandb_logger.experiment.define_metric(f"{step}/acc_epoch", goal="maximize", summary="max,last")
        wandb_logger.experiment.define_metric(f"{step}/f1_epoch", goal="maximize", summary="max,last")

    trainer, pl_model, best_model_path, last_model_path = train(
        experiment,
        model,
        tokenizer,
        dataset,
        res_dir,
        wandb_logger
    )

    if not trainer.is_global_zero:
        return

    trainer = pl.Trainer(
        accelerator='gpu',
        default_root_dir=res_dir,
        devices=1,
        num_nodes=1
    )

    pl_model.load_state_dict(torch.load(last_model_path)['state_dict'])
    model = pl_model.model
    torch.save(model.state_dict(), os.path.join(res_dir, "model_state_dict_last.pt"))
    model.save_pretrained(os.path.join(res_dir, "model_last"))

    val_res = evaluate(experiment, trainer, pl_model, dataset['val'])
    test_res = evaluate(experiment, trainer, pl_model, dataset['test'])

    dump_wandb_summary_metrics(wandb_logger.experiment, val_res, name='val', model='last')
    dump_wandb_summary_metrics(wandb_logger.experiment, test_res, name='test', model='last')

    pl_model.load_state_dict(torch.load(best_model_path)['state_dict'])
    model = pl_model.model
    torch.save(model.state_dict(), os.path.join(res_dir, "model_state_dict_best.pt"))
    model.save_pretrained(os.path.join(res_dir, "model_best"))

    val_res = evaluate(experiment, trainer, pl_model, dataset['val'])
    test_res = evaluate(experiment, trainer, pl_model, dataset['test'])

    dump_wandb_summary_metrics(wandb_logger.experiment, val_res, name='val', model='best')
    dump_wandb_summary_metrics(wandb_logger.experiment, test_res, name='test', model='best')

    wandb_logger.finalize(status='success')


OmegaConf.register_new_resolver(
    "cat", lambda *x: ' '.join(x)
)


@hydra.main(config_path='../../configs', version_base=None)
def main(cfg: DictConfig):
    experiment: Experiment = OmegaConf.merge(OmegaConf.structured(Experiment), cfg)
    seed_everything(experiment.seed)
    logger.info(experiment.wandb.name)

    res_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"{res_dir = }")
    process(experiment, res_dir)


if __name__ == '__main__':
    main()
