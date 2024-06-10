from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.project_dim,
            num_heads=config.attention_heads_num,
            dropout=config.attention_dropout_prob,
            batch_first=True,
        )

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        (
            weighted,
            loss,
            hidden_states,
            attentions,
        ) = super().forward(*args, **kwargs, attention_mask=attention_mask)
        weighted, _ = self.self_attention(
            weighted, weighted, weighted, key_padding_mask=(attention_mask == 0)
        )
        weighted = F.gelu(weighted)
        weighted = F.dropout(
            weighted, p=self.config.attention_dropout_prob, training=self.training
        )
        weighted = (weighted * attention_mask.unsqueeze(-1)).sum(-2) / attention_mask.sum(-1, keepdim=True)
        logits = self.logits(weighted)

        if "labels" in kwargs and kwargs['labels'] is not None:
            loss += F.cross_entropy(logits, kwargs['labels'].view(-1), weight=self.classes_weights)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
