import json
import torch
import os
import gc
import logging

import wandb
from wandb.apis.public import Run

from tqdm.auto import tqdm

from adalayers.training.config import Experiment
from adalayers.training.train import LightningModel
from adalayers.datasets.utils import collapse_tokenized_token_predictions
from adalayers.datasets.conlleval import ids_to_tags, evaluate as conll_evaluate

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)


def dump_wandb_summary_metrics(wandb: Run, results, name, model):
    for k, v in results.items():
        wandb.summary[f"{name}_{model}_{k}"] = v


@torch.no_grad()
def evaluate(experiment: Experiment, model, dataset, dump_file=None):
    torch.cuda.empty_cache()
    gc.collect()

    dataset = dataset.add_column("index", list(range(len(dataset))))
    columns_to_use = ["input_ids", "attention_mask", "index"]
    if 'label' in dataset.column_names:
        columns_to_use.append("label")
    if 'labels' in dataset.column_names:
        columns_to_use.append("labels")

    dataloader = torch.utils.data.DataLoader(
        dataset.select_columns(columns_to_use),
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

    labels_by_rows = []
    predictions_by_rows = []
    logits = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            indexes = batch.pop("index").detach().cpu().numpy().tolist()
            cur_labels = batch["labels"].view(-1).detach().cpu().numpy().tolist()
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(**batch)
            logits += out.logits.detach().cpu().numpy().tolist()
            out = out.logits.argmax(-1).detach().cpu()
            if experiment.optimization.mode == "default":
                labels_by_rows += cur_labels
                predictions_by_rows += out.view(-1).numpy().tolist()
                labels += cur_labels
                predictions += out.view(-1).numpy().tolist()
            elif experiment.optimization.mode == "conll_ner":
                for index, prediction in zip(indexes, out.numpy().tolist()):
                    labels_by_rows.append(ids_to_tags(dataset[index]['ner_tags']))
                    predictions_by_rows.append(
                        ids_to_tags(
                            collapse_tokenized_token_predictions(dataset[index]['word_ids'], prediction)
                        )
                    )
                    labels += labels_by_rows[-1]
                    predictions += predictions_by_rows[-1]
            else:
                raise NotImplementedError()

    if dump_file:
        with open(dump_file, "w") as f:
            f.writelines(
                json.dumps(dict(label=label, prediction=prediction, data=data, logits=logit)) + '\n'
                for label, prediction, data, logit in zip(tqdm(labels_by_rows), predictions_by_rows, dataset, logits)
            )

    if experiment.optimization.mode == "default":
        return dict(
            acc=accuracy_score(labels, predictions, ),
            f1=f1_score(labels, predictions, average="macro"),
            f1_micro=f1_score(labels, predictions, average="micro"),
            recall=recall_score(labels, predictions, average="macro"),
            recall_micro=recall_score(labels, predictions, average="micro"),
            precision=precision_score(labels, predictions, average="macro"),
            precision_micro=precision_score(labels, predictions, average="micro"),
        )
    elif experiment.optimization.mode == "conll_ner":
        prec, rec, f1 = conll_evaluate(labels, predictions)
        return dict(
            acc=accuracy_score(labels, predictions),
            f1_macro=f1_score(labels, predictions, average="macro"),
            precision=prec,
            recall=rec,
            f1=f1,
        )
    else:
        raise NotImplementedError()


def save_wandb_model(run, experiment, directory, name, metrics):
    name = f"{experiment.wandb.name}_{name}"
    name = name.replace(" ", "_")

    metadata = dict(config=experiment, name=name)
    metadata.update(metrics)

    metrics_formatted = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    description = experiment.dataset.name + " | " + metrics_formatted
    artifact = wandb.Artifact(
        name=name, type="model", metadata=metadata, description=description
    )
    artifact.add_dir(local_path=directory)
    run.log_artifact(artifact)


def eval_and_save(
    experiment: Experiment,
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

    save_data = experiment.dataset.save_eval_data

    val_res = evaluate(
        experiment=experiment,
        model=pl_model,
        dataset=dataset["val"],
        dump_file='last_eval.jsonl' if save_data else None
    )
    test_res = evaluate(
        experiment=experiment,
        model=pl_model,
        dataset=dataset["test"],
        dump_file='last_test.jsonl' if save_data else None
    )

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

    val_res = evaluate(
        experiment=experiment,
        model=pl_model,
        dataset=dataset["val"],
        dump_file='best_eval.jsonl' if save_data else None
    )
    test_res = evaluate(
        experiment=experiment,
        model=pl_model,
        dataset=dataset["test"],
        dump_file='best_test.jsonl' if save_data else None
    )

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
