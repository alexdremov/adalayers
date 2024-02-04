import transformers
from adalayers.training.config import Experiment

from adalayers.models.ada_layers_classifier import AdaLayersForSequenceClassification, AdaLayersForSequenceClassificationConfig


def build_model(config: Experiment):
    match config.model.name:
        case "automodel":
            return transformers.AutoModelForSequenceClassification.from_pretrained(
                **config.model.kwargs
            )
        case "adalayers":
            config = AdaLayersForSequenceClassificationConfig(**config.model.kwargs)
            return AdaLayersForSequenceClassification(
                config
            )
        case _:
            raise RuntimeError(f"Unknown model architecture {config.model.name = }")


def build_tokenizer(config: Experiment):
    return transformers.AutoTokenizer.from_pretrained(
        **config.tokenizer_pretrained
    )
