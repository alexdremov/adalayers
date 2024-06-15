from safetensors.torch import load_model
import functools
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
        case "automodel_token":
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                **config.model.kwargs
            )
        case "automodel_freezed":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                **config.model.kwargs
            )
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("classifier")
        case "adalayers":
            config_adalayers = AdaLayersForSequenceClassificationConfig(
                **config.model.kwargs
            )
            model = AdaLayersForSequenceClassification(config_adalayers)
        case "adalayers_token":
            config_adalayers = AdaLayersForTokenClassificationConfig(
                **config.model.kwargs
            )
            model = AdaLayersForTokenClassification(config_adalayers)
        case _:
            raise RuntimeError(f"Unknown model architecture {config.model.name = }")

    if config.model.restore_artifact is not None:
        artifact = run.use_artifact(config.model.restore_artifact)
        if artifact is not None:
            loaded = artifact.download()
            load_model(model, os.path.join(loaded, "model.safetensors"))
        else:
            logger.warning(
                f"No artifact found for {config.model.restore_artifact}. It's fine if this is a DDP subprocess"
            )

    return model


class WrappedTokenizer:
    def __init__(self, tokenizer, call_kwargs, pad_kwargs):
        self.tokenizer = tokenizer
        self.call_kwargs = call_kwargs
        self.pad_kwargs = pad_kwargs

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **{**self.call_kwargs, **kwargs})

    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **{**self.pad_kwargs, **kwargs})

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def build_tokenizer(config: Experiment):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        **{
            k: v
            for k, v in config.tokenizer_pretrained.items()
            if k not in ("tokenize_kwargs", "pad_kwargs")
        }
    )
    return WrappedTokenizer(
        tokenizer=tokenizer,
        call_kwargs=config.tokenizer_pretrained.get("tokenize_kwargs", {}),
        pad_kwargs=config.tokenizer_pretrained.get("pad_kwargs", {}),
    )
