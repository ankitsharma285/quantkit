"""Capture what the first decoder layer receives, without running the rest of the model.

A wrapper replaces layer 0, records its inputs, and raises to stop the forward
pass. The keyword arguments (rotary cos/sin, attention mask, position ids) are
identical for every sample of the same length, so they are stored once.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    def __init__(self, layer: nn.Module, store: list, kwargs_out: dict):
        super().__init__()
        self.layer = layer
        self.store = store
        self.kwargs_out = kwargs_out

    def forward(self, hidden_states, *args, **kwargs):
        if args:
            raise RuntimeError("expected decoder layer extras as keyword arguments")
        self.store.append(hidden_states.detach())
        if not self.kwargs_out:
            self.kwargs_out.update(kwargs)
        raise _Stop


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The ModuleList of decoder blocks (Llama / Qwen / Mistral layout)."""
    try:
        return model.model.layers
    except AttributeError as e:
        raise ValueError("expected a HF causal LM with model.model.layers") from e


@torch.no_grad()
def capture_layer0_inputs(model: nn.Module, tokens: torch.Tensor, device) -> tuple[torch.Tensor, dict]:
    """Run each calibration sample up to layer 0.

    Returns (hidden states [n, seqlen, hidden] in the model dtype, layer kwargs).
    """
    layers = decoder_layers(model)
    store: list[torch.Tensor] = []
    kwargs: dict = {}
    original = layers[0]
    layers[0] = _Catcher(original, store, kwargs)
    try:
        for i in range(tokens.shape[0]):
            try:
                model(tokens[i : i + 1].to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        layers[0] = original
    kwargs.pop("past_key_values", None)
    kwargs["use_cache"] = False
    return torch.cat(store, dim=0), kwargs
