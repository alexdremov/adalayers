import datasets
import logging

from datasets import DownloadMode
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def train_val_split(dataset: datasets.Dataset):
    splitted = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    return dict(
        train=splitted['train'],
        val=splitted['test']
    )


def load_super_glue_rtf(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset('super_glue', 'rte')

    def preprocess(batch):
        batch.update(
            tokenizer(
                batch['premise'], batch['hypothesis'],
                truncation='longest_first',
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset['train'])
    return datasets.DatasetDict(
        train=val_train['train'],
        val=val_train['val'],
        test=dataset['validation'],
        unsupervised=dataset['test']
    )


def load_glue_cola(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset('glue', 'cola')

    def preprocess(batch):
        batch.update(
            tokenizer(
                batch['sentence'],
                truncation=True
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset['train'])
    return datasets.DatasetDict(
        train=val_train['train'],
        val=val_train['val'],
        test=dataset['validation'],
        unsupervised=dataset['test']
    )


def load_imdb(tokenizer: PreTrainedTokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset('imdb')

    def preprocess(batch):
        batch.update(
            tokenizer(
                batch['text'],
                truncation=True
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)
    val_train = train_val_split(dataset['train'])
    return datasets.DatasetDict(
        train=val_train['train'],
        val=val_train['val'],
        test=dataset['test'],
        unsupervised=dataset['unsupervised']
    )


def build_dataset(name, tokenizer):
    match name:
        case 'super_glue_rte':
            return load_super_glue_rtf(tokenizer)
        case 'glue_cola':
            return load_glue_cola(tokenizer)
        case 'imdb':
            return load_imdb(tokenizer)

    logger.error(f'Unknown dataset {name = }')
    raise RuntimeError
