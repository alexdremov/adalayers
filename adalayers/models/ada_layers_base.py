import contextlib
from typing import Optional, Union, Tuple

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import PreTrainedModel

from adalayers.models.configs import AdaLayersBaseConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.register_buffer('pe', self.make_pe(max_len), persistent=False)

    def make_pe(self, max_len: int):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        if seq_len >= len(self.pe):
            self.pe = self.make_pe(seq_len).to(self.pe.device)
        x = x + self.pe[:seq_len][None]
        return self.dropout(x)



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
                    elementwise_affine=False
                )
                for _ in range(config.layers_num)
            ]
        )
        self.pos_emb = PositionalEncoding(
            d_model=config.project_dim,
            dropout=config.attention_dropout_prob,
            max_len=config.layers_num,
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
        assert output_hidden_states is None or output_hidden_states == True, "Must always output hidden states"

        with torch.inference_mode() if self.config.freeze_base_model else contextlib.nullcontext():
            decoder_input_ids = None
            if self.config.generate_fake_decoder_input_ids:
                decoder_input_ids = input_ids[:, :1]
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                decoder_input_ids=decoder_input_ids,
            )

        hiddens = getattr(outputs, "hidden_states", getattr(outputs, "encoder_hidden_states"))

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

        if self.config.add_pos_embeddings:
            weighted = self.pos_emb(weighted)
        return (
            weighted,
            loss,
            hiddens,
            getattr(outputs, "attentions", None),
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
