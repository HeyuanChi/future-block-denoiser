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
    use_initializer: bool = False
    num_diffusion_steps: int = 100
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
    Predicts diffusion noise for future latents from noisy future latents and prefix states.

    Inputs:
        noisy_latent: [B, F, D]
        prefix_states: [B, P, D]
        timesteps: [B]
        prefix_mask: [B, P]
        future_mask: [B, F]
    Output:
        predicted_noise: [B, F, D]
    """

    def __init__(self, config: LatentDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_len = config.coarse_slots or config.future_len

        self.timestep_embedding = nn.Embedding(config.num_diffusion_steps, config.latent_dim)
        self.position_embedding = nn.Embedding(config.prefix_len + self.latent_len, config.latent_dim)
        self.segment_embedding = nn.Embedding(2, config.latent_dim)
        self.input_layer_norm = nn.LayerNorm(config.latent_dim)
        if config.use_initializer:
            self.initializer_queries = nn.Parameter(torch.randn(self.latent_len, config.latent_dim) * 0.02)
            self.initializer_position_embedding = nn.Embedding(config.prefix_len + self.latent_len, config.latent_dim)
            self.initializer_segment_embedding = nn.Embedding(2, config.latent_dim)
            self.initializer_layer_norm = nn.LayerNorm(config.latent_dim)

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

    def initialize_latent(
        self,
        prefix_states: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_initializer:
            raise ValueError("initialize_latent requires use_initializer=True.")

        batch_size, prefix_len, _ = prefix_states.shape
        init_queries = self.initializer_queries.unsqueeze(0).expand(batch_size, -1, -1)
        latent_len = init_queries.size(1)
        token_states = torch.cat([prefix_states, init_queries], dim=1)

        position_ids = torch.arange(prefix_len + latent_len, device=prefix_states.device).unsqueeze(0).expand(
            batch_size,
            prefix_len + latent_len,
        )
        segment_ids = torch.cat(
            [
                torch.zeros(batch_size, prefix_len, device=prefix_states.device, dtype=torch.long),
                torch.ones(batch_size, latent_len, device=prefix_states.device, dtype=torch.long),
            ],
            dim=1,
        )

        token_states = token_states + self.initializer_position_embedding(position_ids)
        token_states = token_states + self.initializer_segment_embedding(segment_ids)
        token_states = self.initializer_layer_norm(token_states)

        latent_mask = torch.ones(batch_size, latent_len, device=prefix_states.device, dtype=prefix_mask.dtype)
        padding_mask = torch.cat([prefix_mask, latent_mask], dim=1) == 0
        token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)
        latent_states = token_states[:, prefix_len:, :]
        return self.output_projection(latent_states)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        prefix_states: torch.Tensor,
        timesteps: torch.Tensor,
        prefix_mask: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, future_len, _ = noisy_latent.shape
        prefix_len = prefix_states.size(1)

        timestep_embed = self.timestep_embedding(timesteps).unsqueeze(1)
        future_states = noisy_latent + timestep_embed

        token_states = torch.cat([prefix_states, future_states], dim=1)

        position_ids = torch.arange(prefix_len + future_len, device=noisy_latent.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, prefix_len + future_len)

        segment_ids = torch.cat(
            [
                torch.zeros(batch_size, prefix_len, device=noisy_latent.device, dtype=torch.long),
                torch.ones(batch_size, future_len, device=noisy_latent.device, dtype=torch.long),
            ],
            dim=1,
        )

        token_states = token_states + self.position_embedding(position_ids) + self.segment_embedding(segment_ids)
        token_states = self.input_layer_norm(token_states)

        padding_mask = torch.cat([prefix_mask, future_mask], dim=1) == 0
        token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)

        future_states = token_states[:, prefix_len:, :]
        return self.output_projection(future_states)
