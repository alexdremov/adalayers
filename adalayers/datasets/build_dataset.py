import datasets
import logging

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


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
    return datasets.DatasetDict(
        train=dataset['train'],
        val=dataset['validation'],
        unsupervised=dataset['test']
    )


def load_glue_cola(tokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset('glue', 'cola')

    def preprocess(batch):
        batch.update(
            tokenizer(
                batch['sentence'],
                truncation='longest_first',
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)

    return datasets.DatasetDict(
        train=dataset['train'],
        val=dataset['validation'],
        unsupervised=dataset['test']
    )


def load_imdb(tokenizer):
    dataset: datasets.DatasetDict = datasets.load_dataset('imdb')

    def preprocess(batch):
        batch.update(
            tokenizer(
                batch['text'],
                truncation='longest_first',
            )
        )
        return batch

    dataset = dataset.map(preprocess, batched=True)

    return datasets.DatasetDict(
        train=dataset['train'],
        val=dataset['test'],
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
