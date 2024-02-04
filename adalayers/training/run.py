import os

import hydra
import hydra.core.hydra_config

import logging
import wandb
import torch

from omegaconf import OmegaConf, DictConfig

from adalayers.training.config import Experiment
from adalayers.models import build_model, build_tokenizer
from adalayers.datasets import build_dataset
from adalayers.training.train import train

logger = logging.getLogger(__name__)


def process(experiment: Experiment, res_dir: str):
    model = build_model(experiment)

    tokenizer = build_tokenizer(experiment)
    dataset = build_dataset(experiment.dataset.name, tokenizer)
    train(
        experiment,
        model,
        tokenizer,
        dataset,
        res_dir
    )

    torch.save(model.state_dict(), os.path.join(res_dir, "model_state_dict.pt"))


OmegaConf.register_new_resolver(
    "cat", lambda *x: ' '.join(x)
)


@hydra.main(config_path='../../configs', version_base=None)
def main(cfg: DictConfig):
    experiment: Experiment = OmegaConf.merge(OmegaConf.structured(Experiment), cfg)
    experiment_resolved = OmegaConf.to_container(
        experiment, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=experiment.wandb.project,
        config=experiment_resolved,
        name=experiment.wandb.name,
        notes=experiment.wandb.notes,
        group="DDP",
    )
    logger.info(experiment)

    res_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"{res_dir = }")
    process(experiment, res_dir)
    wandb.finish()


if __name__ == '__main__':
    main()
