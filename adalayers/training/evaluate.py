import torch
import os
import logging

import wandb
from wandb.apis.public import Run

from tqdm.auto import tqdm

from adalayers.training.config import Experiment
from adalayers.training.train import LightningModel

from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


def dump_wandb_summary_metrics(wandb: Run, results, name, model):
    for k, v in results.items():
        wandb.summary[f"{name}_{model}_{k}"] = v


@torch.no_grad()
def evaluate(experiment: Experiment, model, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset.select_columns(["input_ids", "label", "attention_mask"]),
        batch_size=experiment.optimization.batch_size_eval,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=experiment.optimization.num_workers,
        collate_fn=model.collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "")
    model = model.to(device).eval()
    labels = []
    predictions = []
    for batch in tqdm(dataloader):
        labels += batch["labels"].view(-1).detach().cpu().numpy().tolist()
        for k in batch:
            batch[k] = batch[k].to(device)
        out = model(**batch)
        predictions += out.logits.argmax(-1).view(-1).detach().cpu().numpy().tolist()

    return dict(
        acc=accuracy_score(predictions, labels),
        f1=f1_score(predictions, labels, average="macro"),
    )


def save_wandb_model(run, experiment, directory, name, metrics):
    name = f"{experiment.wandb.name}_{name}"
    name = name.replace(" ", "_")

    metadata = dict(config=experiment, name=name)
    metadata.update(metrics)

    metrics_formatted = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    description = experiment.dataset.name + " | " + metrics_formatted
    artifact = wandb.Artifact(
        name=name, type="model", metadata=metadata, description=description
    )
    artifact.add_dir(local_path=directory)
    run.log_artifact(artifact)


def eval_and_save(
    experiment,
    best_model_path,
    last_model_path,
    dataset,
    model,
    res_dir,
    tokenizer,
    wandb_logger,
):
    pl_model = LightningModel(model, tokenizer, dataset, experiment)
    pl_model.load_state_dict(torch.load(last_model_path)["state_dict"])

    model = pl_model.model
    torch.save(model.state_dict(), os.path.join(res_dir, "model_state_dict_last.pt"))
    model.save_pretrained(os.path.join(res_dir, "model_last"))
    val_res = evaluate(experiment, pl_model, dataset["val"])
    test_res = evaluate(experiment, pl_model, dataset["test"])

    logger.info(f"val metrics for last: {val_res}")
    logger.info(f"test metrics for last: {test_res}")

    save_wandb_model(
        wandb_logger.experiment,
        experiment,
        directory=os.path.join(res_dir, "model_last"),
        name="model_last",
        metrics=test_res,
    )
    dump_wandb_summary_metrics(
        wandb_logger.experiment, val_res, name="val", model="last"
    )
    dump_wandb_summary_metrics(
        wandb_logger.experiment, test_res, name="test", model="last"
    )

    pl_model.load_state_dict(torch.load(best_model_path)["state_dict"])
    model = pl_model.model

    torch.save(model.state_dict(), os.path.join(res_dir, "model_state_dict_best.pt"))
    model.save_pretrained(os.path.join(res_dir, "model_best"))

    val_res = evaluate(experiment, pl_model, dataset["val"])
    test_res = evaluate(experiment, pl_model, dataset["test"])

    logger.info(f"val metrics for best: {val_res}")
    logger.info(f"test metrics for best: {test_res}")

    save_wandb_model(
        wandb_logger.experiment,
        experiment,
        directory=os.path.join(res_dir, "model_best"),
        name="model_best",
        metrics=test_res,
    )
    dump_wandb_summary_metrics(
        wandb_logger.experiment, val_res, name="val", model="best"
    )
    dump_wandb_summary_metrics(
        wandb_logger.experiment, test_res, name="test", model="best"
    )
