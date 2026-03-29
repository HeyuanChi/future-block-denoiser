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
    slot_refinement: str = "none"
    slot_refinement_layers: int = 1
    slot_refinement_scale: float = 0.25
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
        self.slot_refinement = config.slot_refinement

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
        if self.slot_refinement == "causal_residual":
            self.slot_position_embeddings = nn.Embedding(config.coarse_slots, config.latent_dim)
            slot_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.predictor_heads,
                dim_feedforward=config.predictor_ffn_dim,
                dropout=config.predictor_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.slot_refiner = nn.TransformerEncoder(
                encoder_layer=slot_encoder_layer,
                num_layers=config.slot_refinement_layers,
            )
        elif self.slot_refinement != "none":
            raise ValueError(
                f"Unsupported slot_refinement={config.slot_refinement!r}. "
                "Expected 'none' or 'causal_residual'."
            )

    def forward(
        self,
        prefix_states: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        coarse_states = self.output_projection(coarse_states)
        return self.refine_slots(coarse_states)

    def refine_slots(
        self,
        coarse_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.slot_refinement == "none":
            return coarse_states

        batch_size = coarse_states.size(0)
        slot_position_ids = torch.arange(self.config.coarse_slots, device=coarse_states.device).unsqueeze(0).expand(
            batch_size,
            self.config.coarse_slots,
        )
        slot_inputs = coarse_states + self.slot_position_embeddings(slot_position_ids)
        causal_mask = torch.triu(
            torch.full((self.config.coarse_slots, self.config.coarse_slots), float("-inf"), device=coarse_states.device),
            diagonal=1,
        )
        refined_slots = self.slot_refiner(slot_inputs, mask=causal_mask)
        return coarse_states + self.config.slot_refinement_scale * refined_slots
