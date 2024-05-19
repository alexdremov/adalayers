import contextlib
from typing import Optional, Union, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from adalayers.models.configs import AdaLayersBaseConfig


class AdaLayersBase(PreTrainedModel):
    config_class = AdaLayersBaseConfig

    def __init__(self, config: AdaLayersBaseConfig):
        super().__init__(config)
        self.model = transformers.AutoModel.from_pretrained(config.base_model)
        self.model.eval()

        if config.freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.projectors = nn.ModuleList(
            [
                nn.Linear(
                    in_features=config.layer_in_dim,
                    out_features=config.project_dim,
                    bias=False
                )
                for _ in range(config.layers_num)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    normalized_shape=config.project_dim,
                    elementwise_affine=True
                )
                for _ in range(config.layers_num)
            ]
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.project_dim,
            num_heads=config.attention_heads_num,
            dropout=config.attention_dropout_prob,
            batch_first=True,
        )

        if config.classes_weights:
            self.register_buffer("classes_weights", torch.tensor(config.classes_weights), persistent=False)
        else:
            self.classes_weights = None

        distribution = torch.ones(config.layers_num, 1)
        if config.pick_one_layer_only is not None:
            distribution[:] = float("-inf")
            distribution[config.pick_one_layer_only] = 1

        if self.config.freeze_distribution:
            self.register_buffer("distribution", distribution)
        else:
            self.distribution = nn.Parameter(
                distribution,
                requires_grad=True,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ):
        assert output_hidden_states is None or True, "Must always output hidden states"

        with torch.inference_mode() if self.config.freeze_base_model else contextlib.nullcontext():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

        hiddens = outputs.hidden_states

        projected = torch.stack(
            tensors=[
                norm(proj(F.gelu(hidden)))
                for proj, norm, hidden in zip(self.projectors, self.norms, hiddens)
            ],
            dim=-1,
        )

        if self.config.attention_dropout_prob is not None:
            projected = F.dropout(
                projected, p=self.config.attention_dropout_prob, training=self.training
            )

        distribution = self.distribution_normalized

        loss = 0
        if (
            not np.isclose(self.config.lambda_distribution_entropy, 0)
            and not self.config.freeze_distribution
            and labels is not None
        ):
            distribution_entropy = -(
                    distribution * torch.log(distribution + 1e-6)
            ).sum(-1).mean()
            loss += self.config.lambda_distribution_entropy * distribution_entropy

        context = (
            torch.no_grad()
            if self.config.freeze_distribution
            else contextlib.nullcontext()
        )
        with context:
            weighted = F.gelu(projected @ distribution).squeeze(-1)

        weighted, _ = self.self_attention(
            weighted, weighted, weighted, key_padding_mask=(attention_mask == 0)
        )
        weighted = F.gelu(weighted)
        weighted = F.dropout(
            weighted, p=self.config.attention_dropout_prob, training=self.training
        )

        return (
            weighted,
            loss,
            outputs.hidden_states,
            outputs.attentions,
        )

    @property
    def distribution_normalized(self):
        distribution = self.distribution
        if self.config.topk_distribution is not None:
            _, indices_exclude = torch.topk(
                self.distribution,
                k=len(self.distribution) - self.config.topk_distribution,
                dim=0,
                largest=False
            )
            distribution[indices_exclude] = float("-inf")
        return F.softmax(distribution * self.config.alpha_distribution, dim=0)
