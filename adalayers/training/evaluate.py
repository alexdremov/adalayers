import torch
from lightning import Trainer

from adalayers.training.config import Experiment


@torch.no_grad()
def evaluate(experiment: Experiment, trainer: Trainer, model, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset.select_columns(['input_ids', 'label', 'attention_mask']),
        batch_size=experiment.optimization.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=experiment.optimization.num_workers,
        collate_fn=model.collator
    )
    metrics = trainer.test(model, dataloaders=dataloader)[0]
    print(metrics)
    return {
        k.replace('val/', '').replace('_epoch', ''): v
        for k, v in metrics.items()
    }
