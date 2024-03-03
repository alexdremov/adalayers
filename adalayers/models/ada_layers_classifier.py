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


class AdaLayersForSequenceClassification(PreTrainedModel):
    config_class = AdaLayersForSequenceClassificationConfig

    def __init__(self, config: AdaLayersForSequenceClassificationConfig):
        super().__init__(config)
        self.model = transformers.AutoModel.from_pretrained(config.base_model)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.projectors = nn.ModuleList(
            [
                nn.Linear(config.layer_in_dim, config.project_dim, bias=False)
                for _ in range(config.layers_num)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.project_dim) for _ in range(config.layers_num)]
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.project_dim,
            num_heads=config.attention_heads_num,
            dropout=config.attention_dropout_prob,
            batch_first=True,
        )
        self.logits = nn.Linear(config.project_dim, config.num_classes)
        self.distribution = nn.Parameter(
            torch.ones(config.layers_num, 1),
            requires_grad=not self.config.freeze_distribution,
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        assert output_hidden_states is None or True, "Must always output hidden states"

        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

        hiddens = outputs.hidden_states

        projected = torch.stack(
            [
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
            ).mean()
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
        weighted = F.dropout(
            weighted, p=self.config.attention_dropout_prob, training=self.training
        )
        weighted = weighted.mean(-2)
        logits = self.logits(weighted)

        if labels is not None:
            loss += F.cross_entropy(logits, labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def distribution_normalized(self):
        distribution = self.distribution
        if self.config.topk_distribution is not None:
            weights, indices = torch.topk(
                self.distribution, k=self.config.topk_distribution, dim=0
            )
            mask = torch.zeros(self.config.layers_num, device=self.distribution.device)
            mask[indices.view(-1)] = 1

            minus_inf = torch.tensor([float("-inf")], device=self.distribution.device).expand(distribution.shape[0], )
            distribution = torch.where((mask == 1).view(-1), distribution.view(-1), minus_inf.view(-1))[..., None]
        return F.softmax(distribution * self.config.alpha_distribution, dim=0)
