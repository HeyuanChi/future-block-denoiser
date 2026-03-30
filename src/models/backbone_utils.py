from __future__ import annotations

from torch import nn


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
