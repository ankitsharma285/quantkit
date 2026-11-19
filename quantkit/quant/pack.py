"""Int4 packing.

Storage format (this is the contract the Triton kernel reads, so change it here
and nowhere else):

    weight  W      : [N, K]   (out_features, in_features), as in nn.Linear
    qweight        : [K // 8, N] int32
                     eight 4-bit values packed along K; value k = 8*r + i lives
                     in bits [4i, 4i+4) of qweight[r, n]  (little-nibble first)
    scales         : [K // G, N] fp16
    qzeros         : [K // G, N] uint8   (integer zero point, 0..15)

    dequant: W[n, k] = (q[k, n] - qzeros[k // G, n]) * scales[k // G, n]

Packing along K with N contiguous means a kernel tile over (BLOCK_K, BLOCK_N)
loads BLOCK_K // 8 rows of coalesced int32s, the same layout GPTQ/AutoGPTQ use.
Zeros are kept unpacked: they are ~1/G of the weight bytes, and unpacked zeros
keep the kernel simpler.
"""
from __future__ import annotations

import torch

PACK = 8  # int4 values per int32
_SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)  # [0, 4, ..., 28]


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack q: [K, N] integer tensor with values in [0, 15] -> [K // 8, N] int32."""
    if q.dim() != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(q.shape)}")
    K, N = q.shape
    if K % PACK != 0:
        raise ValueError(f"K={K} must be divisible by {PACK}")
    if q.numel() and (q.min() < 0 or q.max() > 15):
        raise ValueError("values must be in [0, 15]")

    q = q.to(torch.int32).view(K // PACK, PACK, N)
    shifts = _SHIFTS.to(q.device).view(1, PACK, 1)
    # Bitwise OR over the 8 nibbles. Nibbles don't overlap, so a sum is equivalent
    # and avoids a Python loop; int32 wraparound on the top nibble is intended.
    return (q << shifts).sum(dim=1, dtype=torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: [K // 8, N] int32 -> [K, N] uint8."""
    if packed.dtype != torch.int32:
        raise ValueError(f"expected int32, got {packed.dtype}")
    R, N = packed.shape
    shifts = _SHIFTS.to(packed.device).view(1, PACK, 1)
    # Arithmetic right shift sign-extends the top nibble; the mask removes it.
    q = (packed.unsqueeze(1) >> shifts) & 0xF
    return q.reshape(R * PACK, N).to(torch.uint8)
