from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        logits: [B, S, V]
        target_ids: [B, S]
        mask: [B, S]
    """
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    loss = loss.view_as(target_ids)
    masked_loss = loss * mask.float()
    normalizer = mask.sum().clamp_min(1).float()
    return masked_loss.sum() / normalizer
