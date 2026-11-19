"""QuantLinear: drop-in nn.Linear replacement holding packed int4 weights.

Day 1 forward = dequantize to the activation dtype, then F.linear. This is the
reference ("dequant-then-matmul") path that the Triton kernel will be tested
against on day 4; the kernel will plug in behind `backend="triton"`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pack import pack_int4, unpack_int4
from .rtn import QuantParams


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        device=None,
    ):
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError("in_features must be divisible by 8 for int4 packing")
        G = in_features if group_size == -1 else group_size
        if in_features % G != 0:
            raise ValueError(f"in_features={in_features} not divisible by group_size={G}")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = G
        self.bits = 4
        self.backend = "reference"
        n_groups = in_features // G

        self.register_buffer(
            "qweight", torch.zeros(in_features // 8, out_features, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "scales", torch.zeros(n_groups, out_features, dtype=dtype, device=device)
        )
        self.register_buffer(
            "qzeros", torch.zeros(n_groups, out_features, dtype=torch.uint8, device=device)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

    @classmethod
    def from_params(
        cls, p: QuantParams, bias: torch.Tensor | None = None, dtype: torch.dtype = torch.float16
    ) -> "QuantLinear":
        if p.bits != 4:
            raise ValueError("QuantLinear stores int4; use fake_quantize for other bit widths")
        N, K = p.q.shape
        m = cls(K, N, p.group_size, bias=bias is not None, dtype=dtype, device=p.q.device)
        # Our kernel layout is K-major: transpose [N, K] -> [K, N] before packing.
        m.qweight.copy_(pack_int4(p.q.t().contiguous()))
        m.scales.copy_(p.scales.t().to(dtype))
        m.qzeros.copy_(p.zeros.t())
        if bias is not None:
            m.bias.copy_(bias.to(dtype))
        return m

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return the dense [N, K] weight this layer represents."""
        dtype = dtype or self.scales.dtype
        K, G = self.in_features, self.group_size
        q = unpack_int4(self.qweight).float()                       # [K, N]
        z = self.qzeros.float().repeat_interleave(G, dim=0)        # [K, N]
        s = self.scales.float().repeat_interleave(G, dim=0)        # [K, N]
        return ((q - z) * s).t().contiguous().to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "reference":
            return F.linear(x, self.dequantize_weight(x.dtype), self.bias)
        raise NotImplementedError(f"backend {self.backend!r} not implemented yet")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, backend={self.backend}"
        )
