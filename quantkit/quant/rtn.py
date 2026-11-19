"""Round-to-nearest (RTN) group-wise quantization.

Shared by every method: GPTQ and AWQ change *which* weights get rounded and how
scales are chosen, but they all end in these same quantize/dequantize ops, so
correctness here is load-bearing for the whole project.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantParams:
    q: torch.Tensor       # [N, K] uint8, values in [0, 2^bits - 1]
    scales: torch.Tensor  # [N, n_groups] float
    zeros: torch.Tensor   # [N, n_groups] uint8
    bits: int
    group_size: int       # resolved (K for per-channel)


def compute_scale_zero(
    w: torch.Tensor, bits: int, sym: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and integer zero point per row of w: [..., G] -> ([...,1], [...,1]).

    Asymmetric: the range [min, max] (always including 0) maps onto [0, qmax].
    Symmetric: zero is fixed at the midpoint and the range is [-absmax, absmax].
    """
    qmax = 2**bits - 1
    w = w.float()
    if sym:
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = 2 * absmax / qmax
        zero = torch.full_like(scale, (qmax + 1) / 2)
    else:
        wmin = w.amin(dim=-1, keepdim=True).clamp(max=0)
        wmax = w.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
        zero = torch.round(-wmin / scale).clamp(0, qmax)
    return scale, zero


def quantize_rtn(
    w: torch.Tensor, bits: int = 4, group_size: int = 128, sym: bool = False
) -> QuantParams:
    """Quantize a weight matrix w: [N, K] with groups of `group_size` along K.

    group_size = -1 means per-output-channel (one group spanning all of K).
    """
    if w.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(w.shape)}")
    if not 2 <= bits <= 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    N, K = w.shape
    G = K if group_size == -1 else group_size
    if K % G != 0:
        raise ValueError(f"K={K} not divisible by group_size={G}")

    qmax = 2**bits - 1
    wg = w.float().reshape(N, K // G, G)
    scale, zero = compute_scale_zero(wg, bits, sym)          # [N, n_groups, 1]
    q = torch.clamp(torch.round(wg / scale) + zero, 0, qmax)
    return QuantParams(
        q=q.reshape(N, K).to(torch.uint8),
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1).to(torch.uint8),
        bits=bits,
        group_size=G,
    )


def dequantize(p: QuantParams, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    N, K = p.q.shape
    G = p.group_size
    q = p.q.float().reshape(N, K // G, G)
    w = (q - p.zeros.float().unsqueeze(-1)) * p.scales.float().unsqueeze(-1)
    return w.reshape(N, K).to(dtype)


def fake_quantize(
    w: torch.Tensor, bits: int = 4, group_size: int = 128, sym: bool = False
) -> torch.Tensor:
    """Quantize then immediately dequantize (same dtype as w). Useful for ablations."""
    return dequantize(quantize_rtn(w, bits, group_size, sym), dtype=w.dtype)
