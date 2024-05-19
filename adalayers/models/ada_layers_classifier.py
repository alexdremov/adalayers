import contextlib
from typing import Optional, Union, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from adalayers.models.configs import AdaLayersForSequenceClassificationConfig
from adalayers.models.ada_layers_base import AdaLayersBase


class AdaLayersForSequenceClassification(AdaLayersBase):
    config_class = AdaLayersForSequenceClassificationConfig

    def __init__(self, config: AdaLayersForSequenceClassificationConfig):
        super().__init__(config)
        self.logits = nn.Linear(
            in_features=config.project_dim,
            out_features=config.num_classes
        )

    def forward(
        self,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        (
            weighted,
            loss,
            hidden_states,
            attentions,
        ) = super().forward(*args, **kwargs)
        weighted = weighted.mean(-2)
        logits = self.logits(weighted)

        if "labels" in kwargs and kwargs['labels'] is not None:
            loss += F.cross_entropy(logits, kwargs['labels'].view(-1), weight=self.classes_weights)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
