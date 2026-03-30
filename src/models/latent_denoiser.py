from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import torch
from torch import nn


@dataclass
class LatentDenoiserConfig:
    latent_dim: int = 256
    prefix_len: int = 64
    future_len: int = 16
    coarse_slots: int | None = None
    num_diffusion_steps: int = 100
    prediction_objective: str = "pred_v"
    self_conditioning: bool = True
    denoiser_layers: int = 4
    denoiser_heads: int = 8
    denoiser_ffn_dim: int = 1024
    denoiser_dropout: float = 0.1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LatentDenoiserConfig":
        model_config = config.get("model", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in model_config.items() if key in valid_keys}
        return cls(**filtered_config)


class LatentDenoiser(nn.Module):
    """
    Predicts a diffusion target from noisy future latents and context states.

    Inputs:
        noisy_latent: [B, F, D]
        context_states: [B, P, D]
        self_condition_latent: [B, F, D] | None
        timesteps: [B]
        context_mask: [B, P]
        future_mask: [B, F]
    Output:
        predicted_target: [B, F, D]
    """

    def __init__(self, config: LatentDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_len = config.coarse_slots or config.future_len

        self.timestep_embedding = nn.Embedding(config.num_diffusion_steps, config.latent_dim)
        self.position_embedding = nn.Embedding(config.prefix_len + self.latent_len, config.latent_dim)
        self.segment_embedding = nn.Embedding(2, config.latent_dim)
        self.input_layer_norm = nn.LayerNorm(config.latent_dim)
        self.self_condition_projection = nn.Linear(config.latent_dim, config.latent_dim)
        self.self_condition_layer_norm = nn.LayerNorm(config.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.denoiser_heads,
            dim_feedforward=config.denoiser_ffn_dim,
            dropout=config.denoiser_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.denoiser_layers,
        )
        self.output_projection = nn.Linear(config.latent_dim, config.latent_dim)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        context_states: torch.Tensor,
        context_mask: torch.Tensor,
        timesteps: torch.Tensor,
        future_mask: torch.Tensor,
        self_condition_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, future_len, _ = noisy_latent.shape
        context_len = context_states.size(1)

        timestep_embed = self.timestep_embedding(timesteps).unsqueeze(1)
        future_states = noisy_latent + timestep_embed
        if self.config.self_conditioning:
            if self_condition_latent is None:
                self_condition_latent = torch.zeros_like(noisy_latent)
            future_states = future_states + self.self_condition_projection(self_condition_latent)
            future_states = self.self_condition_layer_norm(future_states)

        token_states = torch.cat([context_states, future_states], dim=1)

        position_ids = torch.arange(context_len + future_len, device=noisy_latent.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, context_len + future_len)

        segment_ids = torch.cat(
            [
                torch.zeros(batch_size, context_len, device=noisy_latent.device, dtype=torch.long),
                torch.ones(batch_size, future_len, device=noisy_latent.device, dtype=torch.long),
            ],
            dim=1,
        )

        token_states = token_states + self.position_embedding(position_ids) + self.segment_embedding(segment_ids)
        token_states = self.input_layer_norm(token_states)

        padding_mask = torch.cat([context_mask, future_mask], dim=1) == 0
        token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)

        future_states = token_states[:, context_len:, :]
        return self.output_projection(future_states)
