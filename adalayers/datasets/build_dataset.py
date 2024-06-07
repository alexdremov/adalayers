import datasets
import logging

from datasets import DownloadMode
from transformers import PreTrainedTokenizer

from .utils import tokenize_and_align_labels, collapse_tokenized_token_predictions

logger = logging.getLogger(__name__)


def train_val_split(dataset: datasets.Dataset, stratify_by_column="label"):
    splitted = dataset.train_test_split(
        test_size=0.1, seed=42, stratify_by_column=stratify_by_column
    )
    return dict(train=splitted["train"], val=splitted["test"])


def load_super_glue_rte(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset("super_glue", "rte")

    def preprocess(batch):
        batch.update(
            tokenizer(
                [i + "<sep>" + j for i, j in zip(batch["premise"], batch["hypothesis"])],
                truncation=True,
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset["train"])
    return datasets.DatasetDict(
        train=val_train["train"],
        val=val_train["val"],
        test=dataset["validation"],
        unsupervised=dataset["test"],
    )


def load_glue_cola(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset("glue", "cola")

    def preprocess(batch):
        batch.update(tokenizer(batch["sentence"], truncation=True))
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset["train"])
    return datasets.DatasetDict(
        train=val_train["train"],
        val=val_train["val"],
        test=dataset["validation"],
        unsupervised=dataset["test"],
    )


def load_imdb(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset("imdb")

    def preprocess(batch):
        batch.update(tokenizer(batch["text"], truncation=True))
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset["train"])
    return datasets.DatasetDict(
        train=val_train["train"],
        val=val_train["val"],
        test=dataset["test"],
        unsupervised=dataset["unsupervised"],
    )


def load_conll(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset("eriktks/conll2003")

    def preprocess(batch):
        batch.update(
            tokenize_and_align_labels(tokenizer, batch["tokens"], batch["ner_tags"])
        )
        return batch

    dataset = dataset.map(preprocess, batched=False)
    val_train = train_val_split(dataset["train"], stratify_by_column=None)
    return datasets.DatasetDict(
        train=val_train["train"],
        val=val_train["val"],
        test=dataset["test"],
    )


def build_dataset(name, tokenizer):
    match name:
        case "super_glue_rte":
            return load_super_glue_rte(tokenizer)
        case "glue_cola":
            return load_glue_cola(tokenizer)
        case "imdb":
            return load_imdb(tokenizer)
        case "conll":
            return load_conll(tokenizer)

    logger.error(f"Unknown dataset {name = }")
    raise RuntimeError
