from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import TokenClassifierOutput

from adalayers.models.configs import AdaLayersForTokenClassificationConfig
from adalayers.models.ada_layers_base import AdaLayersBase


class AdaLayersForTokenClassification(AdaLayersBase):
    config_class = AdaLayersForTokenClassificationConfig

    def __init__(self, config: AdaLayersForTokenClassificationConfig):
        super().__init__(config)
        self.logits = nn.Linear(
            in_features=config.project_dim,
            out_features=config.num_classes
        )

    def forward(
        self,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        (
            weighted,
            loss,
            hidden_states,
            attentions,
        ) = super().forward(*args, **kwargs)
        logits = self.logits(weighted)

        if "labels" in kwargs and kwargs['labels'] is not None:
            loss += F.cross_entropy(logits.view(-1), kwargs['labels'].view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
