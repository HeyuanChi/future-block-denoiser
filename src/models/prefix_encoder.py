from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class PrefixEncoderConfig:
    bert_name: str = "bert-base-uncased"
    latent_dim: int = 256

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PrefixEncoderConfig":
        model_config = config.get("model", config)
        return cls(
            bert_name=model_config.get("bert_name", "bert-base-uncased"),
            latent_dim=model_config["latent_dim"],
        )


class PrefixEncoder(nn.Module):
    """
    Encodes the prefix tokens into conditioning states.

    Input:
        prefix_ids: [B, P]
        prefix_mask: [B, P]
    Output:
        prefix_states: [B, P, latent_dim]
    """

    def __init__(self, config: PrefixEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.prefix_encoder = AutoModel.from_pretrained(config.bert_name)
        if not hasattr(self.prefix_encoder, "encoder") or not hasattr(self.prefix_encoder.encoder, "layer"):
            raise ValueError(
                f"Backbone {config.bert_name} does not expose encoder.layer; "
                "only BERT/RoBERTa-style encoder backbones are currently supported."
            )
        self.prefix_encoder.encoder.layer = nn.ModuleList(self.prefix_encoder.encoder.layer[:2])
        hidden_size = self.prefix_encoder.config.hidden_size
        self.output_projection = nn.Linear(hidden_size, config.latent_dim)

    def freeze_bert_backbone(self) -> None:
        for parameter in self.prefix_encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.prefix_encoder(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
        )
        return self.output_projection(encoder_outputs.last_hidden_state)
