from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import BertModel


@dataclass
class FutureAutoencoderConfig:
    bert_name: str = "bert-base-uncased"
    future_len: int = 16
    latent_dim: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FutureAutoencoderConfig":
        model_config = config.get("model", config)
        return cls(**model_config)


class FutureAutoencoder(nn.Module):
    """
    Stage 1 future-block autoencoder.

    Input:
        future_ids: [B, 16]
        future_mask: [B, 16]
    Output:
        latent: [B, 16, 256]
        logits: [B, 16, vocab_size]
    """

    def __init__(self, config: FutureAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        bert = BertModel.from_pretrained(config.bert_name)
        self.bert_embeddings = bert.embeddings
        self.future_encoder_layers = nn.ModuleList(bert.encoder.layer[:2])
        hidden_size = bert.config.hidden_size
        vocab_size = bert.config.vocab_size

        self.latent_projection = nn.Linear(hidden_size, config.latent_dim)
        self.latent_to_hidden = nn.Linear(config.latent_dim, hidden_size)
        self.decoder_position_embeddings = nn.Embedding(config.future_len, hidden_size)
        self.decoder_layer_norm = nn.LayerNorm(hidden_size)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=config.decoder_layers,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def freeze_bert_backbone(self) -> None:
        for parameter in self.bert_embeddings.parameters():
            parameter.requires_grad = False
        for layer in self.future_encoder_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def encode_future(
        self,
        future_ids: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            future_ids: [B, S]
            future_mask: [B, S]
        Returns:
            latent: [B, S, latent_dim]
        """
        hidden_states = self.bert_embeddings(input_ids=future_ids)
        extended_mask = future_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min

        for layer in self.future_encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=extended_mask)[0]

        latent = self.latent_projection(hidden_states)
        return latent

    def decode_latent(
        self,
        latent: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent: [B, S, latent_dim]
            future_mask: [B, S]
        Returns:
            logits: [B, S, vocab_size]
        """
        batch_size, seq_len, _ = latent.shape
        hidden_states = self.latent_to_hidden(latent)

        position_ids = torch.arange(seq_len, device=latent.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.decoder_position_embeddings(position_ids)
        hidden_states = self.decoder_layer_norm(hidden_states)

        padding_mask = future_mask == 0
        hidden_states = self.decoder(hidden_states, src_key_padding_mask=padding_mask)
        logits = self.lm_head(hidden_states)
        return logits

    def forward(
        self,
        future_ids: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode_future(future_ids=future_ids, future_mask=future_mask)
        logits = self.decode_latent(latent=latent, future_mask=future_mask)
        return latent, logits
