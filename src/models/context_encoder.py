from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class ContextEncoderConfig:
    bert_name: str = "bert-base-uncased"
    latent_dim: int = 256

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ContextEncoderConfig":
        model_config = config.get("model", config)
        return cls(
            bert_name=model_config.get("bert_name", "bert-base-uncased"),
            latent_dim=model_config["latent_dim"],
        )


class ContextEncoder(nn.Module):
    """
    Encodes a concatenated context sequence into conditioning states.

    Input:
        context_ids: [B, C]
        context_mask: [B, C]
    Output:
        context_states: [B, C, latent_dim]
    """

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.bert_name)
        if not hasattr(self.encoder, "encoder") or not hasattr(self.encoder.encoder, "layer"):
            raise ValueError(
                f"Backbone {config.bert_name} does not expose encoder.layer; "
                "only BERT/RoBERTa-style encoder backbones are currently supported."
            )
        self.encoder.encoder.layer = nn.ModuleList(self.encoder.encoder.layer[:2])
        hidden_size = self.encoder.config.hidden_size
        self.output_projection = nn.Linear(hidden_size, config.latent_dim)

    def freeze_bert_backbone(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=context_ids,
            attention_mask=context_mask,
        )
        return self.output_projection(encoder_outputs.last_hidden_state)
