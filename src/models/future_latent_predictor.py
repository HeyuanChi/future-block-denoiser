from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import torch
from torch import nn


@dataclass
class FutureLatentPredictorConfig:
    latent_dim: int = 256
    prefix_len: int = 64
    coarse_slots: int = 4
    predictor_mode: str = "parallel"
    predictor_layers: int = 2
    predictor_heads: int = 8
    predictor_ffn_dim: int = 1024
    predictor_dropout: float = 0.1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FutureLatentPredictorConfig":
        model_config = config.get("model", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in model_config.items() if key in valid_keys}
        return cls(**filtered_config)


class FutureLatentPredictor(nn.Module):
    """
    Predicts a coarse future latent from prefix states.

    Inputs:
        prefix_states: [B, P, D]
        prefix_mask: [B, P]
    Output:
        coarse_latent: [B, C, D]
    """

    def __init__(self, config: FutureLatentPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.predictor_mode = config.predictor_mode

        self.coarse_queries = nn.Parameter(torch.randn(config.coarse_slots, config.latent_dim) * 0.02)
        self.position_embedding = nn.Embedding(config.prefix_len + config.coarse_slots, config.latent_dim)
        self.segment_embedding = nn.Embedding(2, config.latent_dim)
        self.input_layer_norm = nn.LayerNorm(config.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.predictor_heads,
            dim_feedforward=config.predictor_ffn_dim,
            dropout=config.predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.predictor_layers,
        )
        self.output_projection = nn.Linear(config.latent_dim, config.latent_dim)

        if self.predictor_mode not in {"parallel", "autoregressive"}:
            raise ValueError(
                f"Unsupported predictor_mode={config.predictor_mode!r}. "
                "Expected 'parallel' or 'autoregressive'."
            )

    def forward(
        self,
        prefix_states: torch.Tensor,
        prefix_mask: torch.Tensor,
        target_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.predictor_mode == "autoregressive":
            return self.forward_autoregressive(
                prefix_states=prefix_states,
                prefix_mask=prefix_mask,
                target_latent=target_latent,
            )

        batch_size, prefix_len, _ = prefix_states.shape
        coarse_queries = self.coarse_queries.unsqueeze(0).expand(batch_size, -1, -1)
        coarse_len = coarse_queries.size(1)

        token_states = torch.cat([prefix_states, coarse_queries], dim=1)

        position_ids = torch.arange(prefix_len + coarse_len, device=prefix_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, prefix_len + coarse_len)
        segment_ids = torch.cat(
            [
                torch.zeros(batch_size, prefix_len, device=prefix_states.device, dtype=torch.long),
                torch.ones(batch_size, coarse_len, device=prefix_states.device, dtype=torch.long),
            ],
            dim=1,
        )

        token_states = token_states + self.position_embedding(position_ids) + self.segment_embedding(segment_ids)
        token_states = self.input_layer_norm(token_states)

        coarse_mask = torch.ones(batch_size, coarse_len, device=prefix_states.device, dtype=prefix_mask.dtype)
        padding_mask = torch.cat([prefix_mask, coarse_mask], dim=1) == 0
        token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)

        coarse_states = token_states[:, prefix_len:, :]
        return self.output_projection(coarse_states)

    def forward_autoregressive(
        self,
        prefix_states: torch.Tensor,
        prefix_mask: torch.Tensor,
        target_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, prefix_len, _ = prefix_states.shape
        predicted_slots: list[torch.Tensor] = []

        for slot_idx in range(self.config.coarse_slots):
            if slot_idx == 0:
                previous_slots = prefix_states.new_zeros(batch_size, 0, self.config.latent_dim)
            elif target_latent is not None:
                previous_slots = target_latent[:, :slot_idx, :]
            else:
                previous_slots = torch.cat(predicted_slots, dim=1)

            current_query = self.coarse_queries[slot_idx].view(1, 1, -1).expand(batch_size, 1, -1)
            latent_tokens = torch.cat([previous_slots, current_query], dim=1)
            current_length = prefix_len + latent_tokens.size(1)

            token_states = torch.cat([prefix_states, latent_tokens], dim=1)

            position_ids = torch.arange(current_length, device=prefix_states.device).unsqueeze(0).expand(
                batch_size,
                current_length,
            )
            segment_ids = torch.cat(
                [
                    torch.zeros(batch_size, prefix_len, device=prefix_states.device, dtype=torch.long),
                    torch.ones(batch_size, latent_tokens.size(1), device=prefix_states.device, dtype=torch.long),
                ],
                dim=1,
            )

            token_states = token_states + self.position_embedding(position_ids) + self.segment_embedding(segment_ids)
            token_states = self.input_layer_norm(token_states)

            latent_mask = torch.ones(
                batch_size,
                latent_tokens.size(1),
                device=prefix_states.device,
                dtype=prefix_mask.dtype,
            )
            padding_mask = torch.cat([prefix_mask, latent_mask], dim=1) == 0
            token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)

            current_slot = self.output_projection(token_states[:, -1:, :])
            predicted_slots.append(current_slot)

        return torch.cat(predicted_slots, dim=1)
