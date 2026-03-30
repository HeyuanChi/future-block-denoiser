from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from transformers import AutoModel
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def trim_encoder_layers(backbone: nn.Module, num_layers: int = 2) -> None:
    if not hasattr(backbone, "encoder"):
        raise ValueError("Backbone does not expose an encoder module.")
    encoder = backbone.encoder
    if hasattr(encoder, "layer"):
        encoder.layer = nn.ModuleList(encoder.layer[:num_layers])
        return
    if hasattr(encoder, "layers"):
        encoder.layers = nn.ModuleList(encoder.layers[:num_layers])
        return
    raise ValueError("Backbone encoder does not expose either encoder.layer or encoder.layers.")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(positions).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64) -> None:
        super().__init__()
        heads = max(dim // dim_head, 1)
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.query_norm = RMSNorm(dim_head)
        self.key_norm = RMSNorm(dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = self.heads
        x = self.norm(x)
        query = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=heads)
        key = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=heads)
        value = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=heads)

        similarity = torch.einsum(
            "b h i d, b h j d -> b h i j",
            self.query_norm(query) * self.scale,
            self.key_norm(key),
        )
        attention = similarity.softmax(dim=-1)
        output = torch.einsum("b h i j, b h j d -> b h i d", attention, value)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = max(dim, dim_latent)
        heads = max(inner_dim // dim_head, 1)
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim_latent)
        self.query_norm = RMSNorm(dim_head)
        self.key_norm = RMSNorm(dim_head)
        self.to_q = nn.Linear(dim_latent, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.latent_to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False) if dim_latent != dim else None
        self.to_out = nn.Linear(inner_dim, dim_latent)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        latents = self.norm_latents(latents)
        heads = self.heads

        query = self.to_q(latents)
        x_kv = self.to_kv(x)
        latent_kv = self.latent_to_kv(latents) if self.latent_to_kv is not None else self.to_kv(latents)
        key_value_input = torch.cat([x_kv, latent_kv], dim=1)
        key, value = rearrange(key_value_input, "b n (two h d) -> two b h n d", two=2, h=heads)
        query = rearrange(query, "b n (h d) -> b h n d", h=heads)

        similarity = torch.einsum(
            "b h i d, b h j d -> b h i j",
            self.query_norm(query) * self.scale,
            self.key_norm(key),
        )
        if mask is not None:
            max_neg = -torch.finfo(similarity.dtype).max
            padded_mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            padded_mask = rearrange(padded_mask, "b j -> b 1 1 j")
            similarity = similarity.masked_fill(~padded_mask, max_neg)
        attention = similarity.softmax(dim=-1, dtype=torch.float32).to(similarity.dtype)
        output = torch.einsum("b h i j, b h j d -> b h i d", attention, value)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_latent: int,
        depth: int,
        num_latents: int,
        max_seq_len: int,
        ff_mult: int = 4,
        l2_normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_latent=dim_latent),
                        FeedForward(dim=dim_latent, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim_latent)
        self.l2_normalize_latents = l2_normalize_latents

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.pos_emb(x)
        latents = repeat(self.latents, "n d -> b n d", b=x.size(0))
        for attention, feed_forward in self.layers:
            latents = attention(x, latents, mask=mask) + latents
            latents = feed_forward(latents) + latents
        latents = self.final_norm(latents)
        if self.l2_normalize_latents:
            latents = F.normalize(latents, dim=-1) * math.sqrt(latents.size(-1))
        return latents


class TransformerDecoderBridge(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, depth: int, max_seq_len: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim_input, dim_output)
        self.pos_emb = AbsolutePositionalEmbedding(dim_output, max_seq_len)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim=dim_output),
                        FeedForward(dim=dim_output, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_emb(x)
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return self.final_norm(x)


class PerceiverAutoEncoder(nn.Module):
    def __init__(
        self,
        dim_lm: int,
        dim_ae: int,
        depth: int,
        num_encoder_latents: int,
        num_decoder_latents: int,
        max_seq_len: int,
        l2_normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        self.perceiver_encoder = PerceiverResampler(
            dim=dim_lm,
            dim_latent=dim_ae,
            depth=depth,
            num_latents=num_encoder_latents,
            max_seq_len=max_seq_len,
            l2_normalize_latents=l2_normalize_latents,
        )
        self.perceiver_decoder = TransformerDecoderBridge(
            dim_input=dim_ae,
            dim_output=dim_lm,
            depth=depth,
            max_seq_len=num_decoder_latents,
        )

    def encode(self, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.perceiver_encoder(encoder_outputs, mask=attention_mask.bool())

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.perceiver_decoder(latent)


@dataclass
class FutureAutoencoderConfig:
    bert_name: str = "bert-base-uncased"
    backbone_type: str = "encoder_only"
    future_len: int = 16
    coarse_slots: int = 16
    latent_dim: int = 256
    decoder_latents: int | None = None
    latent_bottleneck_depth: int = 2
    l2_normalize_latents: bool = False
    slot_refinement: str = "none"
    slot_refinement_layers: int = 1
    slot_refinement_scale: float = 0.25
    freeze_backbone: bool = True
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
    """Autoencoder for target text blocks."""

    def __init__(self, config: FutureAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type

        if self.backbone_type == "encoder_only":
            self.future_encoder = AutoModel.from_pretrained(config.bert_name)
            trim_encoder_layers(self.future_encoder, num_layers=2)
            hidden_size = getattr(self.future_encoder.config, "hidden_size", self.future_encoder.config.d_model)
            vocab_size = self.future_encoder.config.vocab_size
            self.seq2seq_backbone = None
        elif self.backbone_type == "bart":
            self.seq2seq_backbone = BartForConditionalGeneration.from_pretrained(config.bert_name)
            self.future_encoder = None
            hidden_size = self.seq2seq_backbone.config.d_model
            vocab_size = self.seq2seq_backbone.config.vocab_size
        else:
            raise ValueError(f"Unsupported backbone_type={self.backbone_type!r}.")
        self.coarse_slots = config.coarse_slots
        self.slot_refinement = config.slot_refinement

        if self.backbone_type == "bart":
            decoder_latents = config.decoder_latents or config.future_len
            self.perceiver_ae = PerceiverAutoEncoder(
                dim_lm=hidden_size,
                dim_ae=config.latent_dim,
                depth=config.latent_bottleneck_depth,
                num_encoder_latents=config.coarse_slots,
                num_decoder_latents=decoder_latents,
                max_seq_len=config.future_len,
                l2_normalize_latents=config.l2_normalize_latents,
            )
            self.latent_projection = None
            self.coarse_queries = None
            self.coarse_cross_attention = None
            self.coarse_layer_norm = None
        else:
            self.perceiver_ae = None
            self.latent_projection = nn.Linear(hidden_size, config.latent_dim)
            self.coarse_queries = nn.Parameter(torch.randn(config.coarse_slots, config.latent_dim) * 0.02)
            self.coarse_cross_attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.decoder_heads,
                batch_first=True,
            )
            self.coarse_layer_norm = nn.LayerNorm(config.latent_dim)
        if self.slot_refinement == "causal_residual":
            self.slot_position_embeddings = nn.Embedding(config.coarse_slots, config.latent_dim)
            slot_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.decoder_heads,
                dim_feedforward=config.decoder_ffn_dim,
                dropout=config.decoder_dropout,
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

        if self.backbone_type == "bart":
            self.expand_queries = None
            self.expand_cross_attention = None
            self.expand_layer_norm = None
        else:
            self.expand_queries = nn.Parameter(torch.randn(config.future_len, config.latent_dim) * 0.02)
            self.expand_cross_attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.decoder_heads,
                batch_first=True,
            )
            self.expand_layer_norm = nn.LayerNorm(config.latent_dim)
        self.latent_to_hidden = nn.Linear(config.latent_dim, hidden_size)
        if self.backbone_type == "encoder_only":
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
            self.decoder_queries = None
        else:
            self.decoder = None
            self.decoder_position_embeddings = None
            self.decoder_layer_norm = None
            self.lm_head = None
            self.decoder_queries = nn.Parameter(torch.randn(config.future_len, hidden_size) * 0.02)

    def freeze_bert_backbone(self) -> None:
        if self.backbone_type == "encoder_only":
            for parameter in self.future_encoder.parameters():
                parameter.requires_grad = False
            return
        for parameter in self.seq2seq_backbone.parameters():
            parameter.requires_grad = False

    def encode_future(
        self,
        future_ids: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone_type == "encoder_only":
            encoder_outputs = self.future_encoder(
                input_ids=future_ids,
                attention_mask=future_mask,
            )
            token_latent = self.latent_projection(encoder_outputs.last_hidden_state)
            return self.compress_latent(token_latent=token_latent, future_mask=future_mask)
        else:
            encoder_outputs = self.seq2seq_backbone.model.encoder(
                input_ids=future_ids,
                attention_mask=future_mask,
                return_dict=True,
            )
            latent = self.perceiver_ae.encode(encoder_outputs.last_hidden_state, future_mask)
            return self.refine_slots(latent)

    def compress_latent(
        self,
        token_latent: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone_type == "bart":
            raise ValueError("compress_latent is only used for encoder_only backbones.")
        if self.coarse_slots == self.config.future_len:
            return token_latent
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
        coarse_latent = self.coarse_layer_norm(coarse_queries + coarse_latent)
        return self.refine_slots(coarse_latent)

    def refine_slots(
        self,
        coarse_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.slot_refinement == "none":
            return coarse_latent

        batch_size = coarse_latent.size(0)
        slot_position_ids = torch.arange(self.coarse_slots, device=coarse_latent.device).unsqueeze(0).expand(
            batch_size,
            self.coarse_slots,
        )
        slot_inputs = coarse_latent + self.slot_position_embeddings(slot_position_ids)
        causal_mask = torch.triu(
            torch.full((self.coarse_slots, self.coarse_slots), float("-inf"), device=coarse_latent.device),
            diagonal=1,
        )
        refined_slots = self.slot_refiner(slot_inputs, mask=causal_mask)
        return coarse_latent + self.config.slot_refinement_scale * refined_slots

    def expand_latent(
        self,
        coarse_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone_type == "bart":
            return coarse_latent
        if self.coarse_slots == self.config.future_len:
            return coarse_latent
        batch_size = coarse_latent.size(0)
        expand_queries = self.expand_queries.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_latent, _ = self.expand_cross_attention(
            query=expand_queries,
            key=coarse_latent,
            value=coarse_latent,
            need_weights=False,
        )
        return self.expand_layer_norm(expand_queries + expanded_latent)

    def decode_latent(
        self,
        latent: torch.Tensor,
        future_mask: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.size(1) != self.config.future_len:
            latent = self.expand_latent(latent)

        if self.backbone_type == "bart":
            return self.decode_latent_with_bart(latent=latent, future_mask=future_mask, target_ids=target_ids)

        batch_size, seq_len, _ = latent.shape
        hidden_states = self.latent_to_hidden(latent)

        position_ids = torch.arange(seq_len, device=latent.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden_states = hidden_states + self.decoder_position_embeddings(position_ids)
        hidden_states = self.decoder_layer_norm(hidden_states)

        padding_mask = future_mask == 0
        hidden_states = self.decoder(hidden_states, src_key_padding_mask=padding_mask)
        logits = self.lm_head(hidden_states)
        return logits

    def decode_latent_with_bart(
        self,
        latent: torch.Tensor,
        future_mask: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = self.perceiver_ae.decode(latent)
        encoder_mask = torch.ones(
            encoder_hidden_states.size(0),
            encoder_hidden_states.size(1),
            device=encoder_hidden_states.device,
            dtype=future_mask.dtype,
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        if target_ids is not None:
            outputs = self.seq2seq_backbone(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_mask,
                labels=target_ids,
                return_dict=True,
            )
            return outputs.logits

        generated_ids = self.seq2seq_backbone.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_mask,
            max_new_tokens=self.config.future_len,
            num_beams=1,
        )
        generated_ids = generated_ids[:, 1:]
        vocab_size = self.seq2seq_backbone.config.vocab_size
        logits = F.one_hot(generated_ids, num_classes=vocab_size).float()
        return logits * 100.0

    def generate_from_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone_type != "bart":
            raise ValueError("generate_from_latent is only supported for the bart backbone.")
        encoder_hidden_states = self.perceiver_ae.decode(latent)
        encoder_mask = torch.ones(
            encoder_hidden_states.size(0),
            encoder_hidden_states.size(1),
            device=encoder_hidden_states.device,
            dtype=torch.long,
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        generated_ids = self.seq2seq_backbone.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_mask,
            max_new_tokens=self.config.future_len,
            num_beams=1,
        )
        return generated_ids[:, 1:]

    def forward(
        self,
        future_ids: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode_future(future_ids=future_ids, future_mask=future_mask)
        logits = self.decode_latent(latent=latent, future_mask=future_mask, target_ids=future_ids)
        return latent, logits
