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
    predictor_mode: str = "deterministic"
    plan_latent_dim: int = 256
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
        self.predictor_mode = config.predictor_mode
        self.slot_refinement = config.slot_refinement

        self.coarse_queries = nn.Parameter(torch.randn(config.coarse_slots, config.latent_dim) * 0.02)
        self.position_embedding = nn.Embedding(config.prefix_len + config.coarse_slots + 1, config.latent_dim)
        self.segment_embedding = nn.Embedding(3, config.latent_dim)
        self.input_layer_norm = nn.LayerNorm(config.latent_dim)
        self._last_kl_loss: torch.Tensor | None = None

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
        if self.predictor_mode == "cvae":
            self.prior_network = nn.Sequential(
                nn.Linear(config.latent_dim, config.predictor_ffn_dim),
                nn.GELU(),
                nn.Linear(config.predictor_ffn_dim, 2 * config.plan_latent_dim),
            )
            self.posterior_network = nn.Sequential(
                nn.Linear(2 * config.latent_dim, config.predictor_ffn_dim),
                nn.GELU(),
                nn.Linear(config.predictor_ffn_dim, 2 * config.plan_latent_dim),
            )
            self.plan_projection = nn.Linear(config.plan_latent_dim, config.latent_dim)
        elif self.predictor_mode != "deterministic":
            raise ValueError(
                f"Unsupported predictor_mode={config.predictor_mode!r}. "
                "Expected 'deterministic' or 'cvae'."
            )
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
        target_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, prefix_len, _ = prefix_states.shape
        plan_token, posterior_used = self.build_plan_token(
            prefix_states=prefix_states,
            prefix_mask=prefix_mask,
            target_latent=target_latent,
        )
        coarse_queries = self.coarse_queries.unsqueeze(0).expand(batch_size, -1, -1)
        coarse_len = coarse_queries.size(1)

        token_parts = [prefix_states]
        segment_parts = [
            torch.zeros(batch_size, prefix_len, device=prefix_states.device, dtype=torch.long),
        ]
        if plan_token is not None:
            token_parts.append(plan_token)
            segment_parts.append(torch.ones(batch_size, 1, device=prefix_states.device, dtype=torch.long))
        token_parts.append(coarse_queries)
        segment_parts.append(
            torch.full(
                (batch_size, coarse_len),
                2 if plan_token is not None else 1,
                device=prefix_states.device,
                dtype=torch.long,
            )
        )
        token_states = torch.cat(token_parts, dim=1)
        total_len = token_states.size(1)

        position_ids = torch.arange(total_len, device=prefix_states.device).unsqueeze(0).expand(batch_size, total_len)
        segment_ids = torch.cat(segment_parts, dim=1)

        token_states = token_states + self.position_embedding(position_ids) + self.segment_embedding(segment_ids)
        token_states = self.input_layer_norm(token_states)

        coarse_mask = torch.ones(batch_size, coarse_len, device=prefix_states.device, dtype=prefix_mask.dtype)
        mask_parts = [prefix_mask]
        if plan_token is not None:
            mask_parts.append(torch.ones(batch_size, 1, device=prefix_states.device, dtype=prefix_mask.dtype))
        mask_parts.append(coarse_mask)
        padding_mask = torch.cat(mask_parts, dim=1) == 0
        token_states = self.transformer(token_states, src_key_padding_mask=padding_mask)

        coarse_states = token_states[:, -coarse_len:, :]
        coarse_states = self.output_projection(coarse_states)
        if not posterior_used:
            self._last_kl_loss = coarse_states.new_zeros(())
        return self.refine_slots(coarse_states)

    def build_plan_token(
        self,
        prefix_states: torch.Tensor,
        prefix_mask: torch.Tensor,
        target_latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, bool]:
        if self.predictor_mode == "deterministic":
            self._last_kl_loss = prefix_states.new_zeros(())
            return None, False

        prefix_summary = self.masked_mean(prefix_states, prefix_mask)
        prior_mean, prior_logvar = self.split_gaussian_params(self.prior_network(prefix_summary))

        use_posterior = self.training and target_latent is not None
        if use_posterior:
            target_summary = target_latent.mean(dim=1)
            posterior_input = torch.cat([prefix_summary, target_summary], dim=-1)
            posterior_mean, posterior_logvar = self.split_gaussian_params(self.posterior_network(posterior_input))
            plan_latent = self.reparameterize(posterior_mean, posterior_logvar)
            self._last_kl_loss = self.compute_kl_divergence(
                posterior_mean=posterior_mean,
                posterior_logvar=posterior_logvar,
                prior_mean=prior_mean,
                prior_logvar=prior_logvar,
            )
        else:
            plan_latent = prior_mean
            self._last_kl_loss = prefix_states.new_zeros(())

        return self.plan_projection(plan_latent).unsqueeze(1), use_posterior

    def masked_mean(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def split_gaussian_params(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.chunk(params, chunks=2, dim=-1)
        return mean, logvar.clamp(min=-10.0, max=10.0)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def compute_kl_divergence(
        self,
        posterior_mean: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        kl = 0.5 * (
            prior_logvar
            - posterior_logvar
            + (torch.exp(posterior_logvar) + (posterior_mean - prior_mean).pow(2)) / torch.exp(prior_logvar)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

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

    def get_last_kl_loss(self) -> torch.Tensor:
        if self._last_kl_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._last_kl_loss
