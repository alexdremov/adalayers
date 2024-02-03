import hydra
import logging
import wandb
import torch

from omegaconf import OmegaConf, DictConfig

from adalayers.training.config import Experiment
from adalayers.models import build_model, build_tokenizer
from adalayers.datasets import build_dataset
from adalayers.training.train import train

logger = logging.getLogger(__name__)


def process(experiment: Experiment):
    model = build_model(experiment)
    wandb.watch(model)

    tokenizer = build_tokenizer(experiment)
    dataset = build_dataset(experiment.dataset.name, tokenizer)
    train(
        experiment,
        model,
        tokenizer,
        dataset
    )

OmegaConf.register_new_resolver(
    "cat", lambda *x: ' '.join(x)
)


@hydra.main(config_path='../../configs', version_base=None, config_name='base_exp')
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
    process(experiment)
    wandb.finish()


if __name__ == '__main__':
    main()
