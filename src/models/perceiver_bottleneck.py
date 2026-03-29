from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn


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
        h = self.heads
        x = self.norm(x)
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        sim = einsum("b h i d, b h j d -> b h i j", self.query_norm(q) * self.scale, self.key_norm(k))
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


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
        h = self.heads

        q = self.to_q(latents)
        x_kv = self.to_kv(x)
        latent_kv = self.latent_to_kv(latents) if self.latent_to_kv is not None else self.to_kv(latents)
        kv_input = torch.cat([x_kv, latent_kv], dim=1)
        k, v = rearrange(kv_input, "b n (two h d) -> two b h n d", two=2, h=h)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        sim = einsum("b h i d, b h j d -> b h i j", self.query_norm(q) * self.scale, self.key_norm(k))
        if mask is not None:
            max_neg = -torch.finfo(sim.dtype).max
            padded_mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            padded_mask = rearrange(padded_mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~padded_mask, max_neg)
        attn = sim.softmax(dim=-1, dtype=torch.float32).to(sim.dtype)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


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
        for attn, ff in self.layers:
            latents = attn(x, latents, mask=mask) + latents
            latents = ff(latents) + latents
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
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
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
