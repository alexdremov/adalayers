import transformers
from adalayers.training.config import Experiment


def build_model(config: Experiment):
    match config.model.name:
        case "automodel":
            return transformers.AutoModelForSequenceClassification.from_pretrained(
                **config.model.kwargs
            )
        case _:
            raise RuntimeError(f"Unknown model architecture {config.model.name = }")


def build_tokenizer(config: Experiment):
    return transformers.AutoTokenizer.from_pretrained(
        **config.tokenizer_pretrained
    )
