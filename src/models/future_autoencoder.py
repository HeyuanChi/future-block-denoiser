from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class FutureAutoencoderConfig:
    bert_name: str = "bert-base-uncased"
    future_len: int = 16
    coarse_slots: int = 16
    latent_dim: int = 256
    latent_structure: str = "attention"
    latent_conv_kernel_size: int = 3
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FutureAutoencoderConfig":
        model_config = config.get("model", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in model_config.items() if key in valid_keys}
        return cls(**filtered_config)


class FutureAutoencoder(nn.Module):
    """
    Stage 1 future-block autoencoder.

    Input:
        future_ids: [B, 16]
        future_mask: [B, 16]
    Output:
        latent: [B, coarse_slots, 256]
        logits: [B, 16, vocab_size]
    """

    def __init__(self, config: FutureAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        self.future_encoder = AutoModel.from_pretrained(config.bert_name)
        if not hasattr(self.future_encoder, "encoder") or not hasattr(self.future_encoder.encoder, "layer"):
            raise ValueError(
                f"Backbone {config.bert_name} does not expose encoder.layer; "
                "only BERT/RoBERTa-style encoder backbones are currently supported."
            )
        self.future_encoder.encoder.layer = nn.ModuleList(self.future_encoder.encoder.layer[:2])
        hidden_size = self.future_encoder.config.hidden_size
        vocab_size = self.future_encoder.config.vocab_size
        self.coarse_slots = config.coarse_slots
        self.latent_structure = config.latent_structure
        self.downsample_factor = max(config.future_len // max(config.coarse_slots, 1), 1)
        self.use_conv_compress = self.latent_structure in {"conv_pool", "conv_compress_attention_expand"}
        self.use_attention_expand = self.latent_structure in {"attention", "conv_compress_attention_expand"}

        self.latent_projection = nn.Linear(hidden_size, config.latent_dim)
        if self.latent_structure in {"attention", "conv_compress_attention_expand"}:
            self.coarse_queries = nn.Parameter(torch.randn(config.coarse_slots, config.latent_dim) * 0.02)
            self.coarse_cross_attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.decoder_heads,
                batch_first=True,
            )
            self.expand_queries = nn.Parameter(torch.randn(config.future_len, config.latent_dim) * 0.02)
            self.expand_cross_attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.decoder_heads,
                batch_first=True,
            )
        if self.use_conv_compress:
            if config.future_len % config.coarse_slots != 0:
                raise ValueError(
                    "Structured latent compression requires future_len to be divisible by coarse_slots."
                )
            padding = config.latent_conv_kernel_size // 2
            self.downsample_conv = nn.Conv1d(
                in_channels=config.latent_dim,
                out_channels=config.latent_dim,
                kernel_size=config.latent_conv_kernel_size,
                padding=padding,
            )
        if self.latent_structure == "conv_pool":
            self.upsample_deconv = nn.ConvTranspose1d(
                in_channels=config.latent_dim,
                out_channels=config.latent_dim,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            )
            self.upsample_refine = nn.Sequential(
                nn.GELU(),
                nn.Conv1d(
                    in_channels=config.latent_dim,
                    out_channels=config.latent_dim,
                    kernel_size=config.latent_conv_kernel_size,
                    padding=padding,
                ),
            )
        elif self.latent_structure not in {"attention", "conv_compress_attention_expand"}:
            raise ValueError(
                f"Unsupported latent_structure={config.latent_structure!r}. "
                "Expected 'attention', 'conv_pool', or 'conv_compress_attention_expand'."
            )
        self.coarse_layer_norm = nn.LayerNorm(config.latent_dim)
        self.expand_layer_norm = nn.LayerNorm(config.latent_dim)
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
        for parameter in self.future_encoder.parameters():
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
            latent: [B, coarse_slots, latent_dim]
        """
        encoder_outputs = self.future_encoder(
            input_ids=future_ids,
            attention_mask=future_mask,
        )
        token_latent = self.latent_projection(encoder_outputs.last_hidden_state)
        return self.compress_latent(token_latent=token_latent, future_mask=future_mask)

    def compress_latent(
        self,
        token_latent: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.coarse_slots == self.config.future_len:
            return token_latent
        if self.use_conv_compress:
            return self.compress_latent_conv_pool(token_latent=token_latent, future_mask=future_mask)
        batch_size = token_latent.size(0)
        coarse_queries = self.coarse_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = future_mask == 0
        coarse_latent, _ = self.coarse_cross_attention(
            query=coarse_queries,
            key=token_latent,
            value=token_latent,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.coarse_layer_norm(coarse_queries + coarse_latent)

    def compress_latent_conv_pool(
        self,
        token_latent: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, latent_dim = token_latent.shape
        block_size = self.downsample_factor

        smoothed_latent = self.downsample_conv(token_latent.transpose(1, 2)).transpose(1, 2)
        smoothed_latent = smoothed_latent.masked_fill(future_mask.unsqueeze(-1) == 0, 0.0)

        block_latent = smoothed_latent.reshape(batch_size, self.coarse_slots, block_size, latent_dim)
        block_mask = future_mask.reshape(batch_size, self.coarse_slots, block_size, 1).to(block_latent.dtype)
        block_sum = (block_latent * block_mask).sum(dim=2)
        block_count = block_mask.sum(dim=2).clamp_min(1.0)
        coarse_latent = block_sum / block_count
        return self.coarse_layer_norm(coarse_latent)

    def expand_latent(
        self,
        coarse_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.coarse_slots == self.config.future_len:
            return coarse_latent
        if not self.use_attention_expand:
            return self.expand_latent_conv_pool(coarse_latent=coarse_latent)
        batch_size = coarse_latent.size(0)
        expand_queries = self.expand_queries.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_latent, _ = self.expand_cross_attention(
            query=expand_queries,
            key=coarse_latent,
            value=coarse_latent,
            need_weights=False,
        )
        return self.expand_layer_norm(expand_queries + expanded_latent)

    def expand_latent_conv_pool(
        self,
        coarse_latent: torch.Tensor,
    ) -> torch.Tensor:
        repeated_latent = coarse_latent.repeat_interleave(self.downsample_factor, dim=1)
        upsampled_latent = self.upsample_deconv(coarse_latent.transpose(1, 2)).transpose(1, 2)
        upsampled_latent = self.upsample_refine(upsampled_latent.transpose(1, 2)).transpose(1, 2)
        return self.expand_layer_norm(repeated_latent + upsampled_latent)

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
        if latent.size(1) != self.config.future_len:
            latent = self.expand_latent(latent)

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
