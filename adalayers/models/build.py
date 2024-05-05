from safetensors.torch import load_model
import transformers
import os
import logging
from adalayers.training.config import Experiment

from adalayers.models.ada_layers_classifier import (
    AdaLayersForSequenceClassification,
    AdaLayersForSequenceClassificationConfig,
)

from adalayers.models.ada_layers_token_classifier import (
    AdaLayersForTokenClassification,
    AdaLayersForTokenClassificationConfig,
)

logger = logging.getLogger(__name__)

def build_model(config: Experiment, run):
    match config.model.name:
        case "automodel":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                **config.model.kwargs
            )
        case "automodel_freezed":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                **config.model.kwargs
            )
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("classifier")
        case "adalayers":
            config_adalayers = AdaLayersForSequenceClassificationConfig(**config.model.kwargs)
            model = AdaLayersForSequenceClassification(config_adalayers)
        case "adalayers_token":
            config_adalayers = AdaLayersForTokenClassificationConfig(**config.model.kwargs)
            model = AdaLayersForTokenClassification(config_adalayers)
        case _:
            raise RuntimeError(f"Unknown model architecture {config.model.name = }")

    if config.model.restore_artifact is not None:
        artifact = run.use_artifact(config.model.restore_artifact)
        if artifact is not None:
            loaded = artifact.download()
            load_model(model, os.path.join(loaded, 'model.safetensors'))
        else:
            logger.warning(f"No artifact found for {config.model.restore_artifact}. It's fine if this is a DDP subprocess")

    return model


def build_tokenizer(config: Experiment):
    return transformers.AutoTokenizer.from_pretrained(**config.tokenizer_pretrained)
