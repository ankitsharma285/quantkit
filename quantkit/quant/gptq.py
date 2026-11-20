"""GPTQ (Frantar et al., 2022): quantize a Linear while minimizing its output error.

RTN minimizes each weight's own rounding error. GPTQ minimizes the layer's
*output* error ||WX - QX||^2 over calibration inputs X. Quantizing columns
(input channels) one at a time, the optimal correction for the error made on
column j is spread over the not-yet-quantized columns via the inverse Hessian
H = 2 X X^T (Optimal Brain Surgeon update):

    err_j  = (w_j - q_j) / [H^-1]_jj
    W_rest -= err_j * [H^-1]_{j, rest}

Two tricks make this fast and stable; both are exact rewrites, not approximations:

1. Cholesky: the sequence of H^-1 "remove column j" updates equals reading rows
   of the upper Cholesky factor of H^-1, so H^-1 is factorized once.
2. Lazy blocks: updates within a block of `blocksize` columns are applied at
   once; the rest of W is updated once per block (a GEMM instead of K rank-1s).

tests/test_gptq.py checks this implementation against a naive column-by-column
OBS loop with explicit inverse updates.
"""
from __future__ import annotations

import torch

from .rtn import QuantParams, compute_scale_zero


class HessianAccumulator:
    """Running H = (2/n) * sum_i x_i x_i^T over all calibration tokens x_i: [K]."""

    def __init__(self, in_features: int, device=None):
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n = 0

    @torch.no_grad()
    def add_batch(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(self.H.dtype)  # [tokens, K]
        t = x.shape[0]
        # Rescale the old average and add the new tokens; keeps H at O(1) magnitude.
        self.H *= self.n / (self.n + t)
        self.n += t
        x = x * (2.0 / self.n) ** 0.5
        self.H.addmm_(x.t(), x)


def _scale_zero(w: torch.Tensor, bits: int, sym: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row scale/zero for a group, with the scale rounded to fp16 up front.

    QuantLinear stores scales in fp16. GPTQ's error feedback depends on the exact
    quantized values, so it must quantize with the scale that will be stored,
    not an fp32 scale that is rounded later.
    """
    s, z = compute_scale_zero(w, bits, sym)
    return s.half().float().clamp(min=6e-8), z


def _resolve_group(K: int, group_size: int, blocksize: int) -> int:
    G = K if group_size == -1 else group_size
    if K % G != 0:
        raise ValueError(f"K={K} not divisible by group_size={G}")
    if not (G % blocksize == 0 or blocksize % G == 0):
        raise ValueError(f"group_size={G} and blocksize={blocksize} must divide one another")
    return G


@torch.no_grad()
def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
    static_groups: bool = False,
) -> tuple[QuantParams, dict]:
    """Quantize W: [N, K] given H: [K, K]. Returns (params, stats).

    Group scales are either dynamic (default, as in the GPTQ paper: computed when
    a group starts, from the weights after earlier error feedback) or static
    (computed up front from the original weights). act_order processes columns in
    decreasing diag(H) and always uses static groups, so groups stay contiguous
    in K and the storage format and kernel need no per-column group index.

    stats["scale_growth"] = mean final scale / mean RTN scale on the original
    weights. Well above 1 means error feedback inflated the weights before their
    group scale was chosen, coarsening the grid for the whole group.
    """
    if W.dim() != 2 or H.shape != (W.shape[1], W.shape[1]):
        raise ValueError(f"shape mismatch: W {tuple(W.shape)}, H {tuple(H.shape)}")
    N, K = W.shape
    G = _resolve_group(K, group_size, blocksize)
    n_groups = K // G
    qmax = 2**bits - 1

    # fp32 normally; fp64 if the caller passes fp64 (used by the reference test).
    work = torch.float64 if W.dtype == torch.float64 else torch.float32
    W = W.clone().to(work)
    H = H.clone().to(work)

    # Input channels that never fire carry no information: zero them and make H invertible.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    static = act_order or static_groups
    scales = torch.zeros(N, n_groups, dtype=work, device=W.device)
    zeros = torch.zeros(N, n_groups, dtype=work, device=W.device)
    for g in range(n_groups):
        s, z = _scale_zero(W[:, g * G : (g + 1) * G], bits, sym)
        scales[:, g], zeros[:, g] = s.squeeze(1), z.squeeze(1)
    rtn_scale_mean = float(scales.mean())
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        perm = torch.arange(K, device=W.device)

    hdiag = torch.diag(H)
    hdiag_stats = {
        "hdiag_max": float(hdiag.max()),
        "hdiag_median": float(hdiag.median()),
        "hdiag_mean": float(hdiag.mean()),
    }
    damp = percdamp * torch.mean(torch.diag(H))
    H.diagonal().add_(damp)
    # Upper Cholesky factor of H^-1: row j holds the OBS update for column j.
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    losses = torch.zeros(N, dtype=work, device=W.device)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(i2 - i1):
            col = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]

            if act_order:
                g = int(perm[col]) // G
            else:
                g = col // G
                if col % G == 0 and not static:
                    # Scales come from the weights as they are *now*, after the error
                    # feedback from every earlier column has been applied.
                    cur = W[:, col : col + G] if i == 0 else W1[:, i : i + G]
                    s, z = _scale_zero(cur, bits, sym)
                    scales[:, g], zeros[:, g] = s.squeeze(1), z.squeeze(1)
            s, z = scales[:, g], zeros[:, g]

            q = (torch.clamp(torch.round(w / s) + z, 0, qmax) - z) * s
            Q1[:, i] = q
            losses += (w - q) ** 2 / d**2 / 2
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if act_order:
        Q = Q[:, invperm]

    # Recover integer codes from the dequantized values; exact up to float rounding.
    s_full = scales.repeat_interleave(G, dim=1)
    z_full = zeros.repeat_interleave(G, dim=1)
    q_int = torch.clamp(torch.round(Q / s_full + z_full), 0, qmax)

    params = QuantParams(
        q=q_int.to(torch.uint8),
        scales=scales.float(),
        zeros=zeros.to(torch.uint8),
        bits=bits,
        group_size=G,
    )
    stats = {"loss": float(losses.sum()), "damp": float(damp), "dead_cols": int(dead.sum()),
             "scale_growth": float(scales.mean()) / max(rtn_scale_mean, 1e-30), **hdiag_stats}
    return params, stats
