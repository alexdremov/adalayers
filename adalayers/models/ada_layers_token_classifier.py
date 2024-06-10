from typing import Union, Tuple, Optional

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
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.config.attention_layers_num):
            self.encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=config.project_dim,
                    nhead=config.attention_heads_num,
                    dropout=config.attention_dropout_prob,
                    dim_feedforward=self.config.dim_feedforward,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
            )

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        (
            weighted,
            loss,
            hidden_states,
            attentions,
        ) = super().forward(*args, **kwargs, attention_mask=attention_mask)
        key_padding_mask=(attention_mask == 0)
        for module in self.encoder_layers:
            weighted = module(weighted, src_key_padding_mask=key_padding_mask)
        weighted = F.dropout(
            weighted, p=self.config.attention_dropout_prob, training=self.training
        )
        logits = self.logits(weighted)

        if "labels" in kwargs and kwargs['labels'] is not None:
            ce_loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=kwargs['labels'].view(-1),
                reduction='none' if self.config.focal_loss_enabled else 'mean',
                ignore_index=-100,
                weight=self.classes_weights,
            )
            if self.config.focal_loss_enabled:
                pt = torch.exp(-ce_loss)
                alpha, gamma = self.config.focal_loss_alpha, self.config.focal_loss_gamma
                ce_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
            loss += ce_loss

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
