"""Fuse q/k/v and gate/up projections into single matmuls, at runtime.

At decode each projection is a separate small matmul; k_proj and v_proj on
Llama-3.2-1B (N = 512) cannot fill the GPU in one call (see the Phase 5
profile). Concatenating them along the output dimension turns three calls into
one with N = 3072, and gate/up into one with N = 16384.

The HF Llama modules call q_proj, k_proj, v_proj (and gate_proj, up_proj) in
that order on the same input tensor. Each original projection is replaced by a
slice view of one shared fused module: the first call computes the fused output
and remembers its input; the later calls return their slice if handed that same
tensor object, and otherwise recompute, so the result is correct whatever the
calling code does. Works for nn.Linear (fp16 baseline) and QuantLinear (int4,
any backend). This is a runtime transform; fused models are not saved.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .quant.qlinear import QuantLinear


class _FusedGroup(nn.Module):
    def __init__(self, fused: nn.Module):
        super().__init__()
        self.fused = fused
        self._last_x = None
        self._last_y = None

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fused(x)
        self._last_x, self._last_y = x, y
        return y

    def cached(self, x: torch.Tensor) -> torch.Tensor:
        return self._last_y if x is self._last_x else self.compute(x)


class _Slice(nn.Module):
    """Stands in for one original projection; returns its columns of the fused output."""

    def __init__(self, group: _FusedGroup, start: int, end: int, first: bool):
        super().__init__()
        object.__setattr__(self, "_group", group)  # not a submodule: owned by the parent
        self.start, self.end, self.first = start, end, first
        self.out_features = end - start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._group.compute(x) if self.first else self._group.cached(x)
        return y[..., self.start : self.end]

    def extra_repr(self) -> str:
        return f"columns [{self.start}, {self.end}) of fused output"


def _concat(mods: list[nn.Module]) -> nn.Module:
    if all(isinstance(m, QuantLinear) for m in mods):
        K, G = mods[0].in_features, mods[0].group_size
        if any(m.in_features != K or m.group_size != G for m in mods):
            raise ValueError("fused QuantLinears need equal in_features and group_size")
        has_bias = any(m.bias is not None for m in mods)
        f = QuantLinear(K, sum(m.out_features for m in mods), G, bias=has_bias,
                        dtype=mods[0].scales.dtype, device=mods[0].qweight.device)
        f.qweight.copy_(torch.cat([m.qweight for m in mods], dim=1))
        f.scales.copy_(torch.cat([m.scales for m in mods], dim=1))
        f.qzeros.copy_(torch.cat([m.qzeros for m in mods], dim=1))
        if has_bias:
            f.bias.copy_(torch.cat([m.bias if m.bias is not None else torch.zeros_like(m.scales[0])
                                    for m in mods]))
        f.backend = mods[0].backend
        return f
    if all(isinstance(m, nn.Linear) for m in mods):
        w = torch.cat([m.weight.data for m in mods], dim=0)
        has_bias = any(m.bias is not None for m in mods)
        f = nn.Linear(w.shape[1], w.shape[0], bias=has_bias, device=w.device, dtype=w.dtype)
        f.weight.data.copy_(w)
        if has_bias:
            f.bias.data.copy_(torch.cat([m.bias.data if m.bias is not None
                                         else torch.zeros(m.out_features, device=w.device, dtype=w.dtype)
                                         for m in mods]))
        return f
    raise TypeError("can only fuse all-nn.Linear or all-QuantLinear groups")


def _fuse_group(parent: nn.Module, names: list[str], group_attr: str) -> None:
    mods = [getattr(parent, n) for n in names]
    group = _FusedGroup(_concat(mods))
    setattr(parent, group_attr, group)
    start = 0
    for i, (n, m) in enumerate(zip(names, mods)):
        setattr(parent, n, _Slice(group, start, start + m.out_features, first=(i == 0)))
        start += m.out_features


def fuse_projections(model: nn.Module, qkv: bool = True, gate_up: bool = True) -> int:
    """Fuse q/k/v and gate/up in every decoder layer, in place. Returns the number of fused groups."""
    n = 0
    for layer in model.model.layers:
        if qkv:
            _fuse_group(layer.self_attn, ["q_proj", "k_proj", "v_proj"], "qkv_fused")
            n += 1
        if gate_up:
            _fuse_group(layer.mlp, ["gate_proj", "up_proj"], "gate_up_fused")
            n += 1
    return n
