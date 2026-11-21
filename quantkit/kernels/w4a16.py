"""Fused W4A16 matmul in Triton: out[M, N] = x[M, K] @ dequant(W)[K, N].

Weights stay packed in global memory (see quant/pack.py) and are dequantized in
registers inside the matmul loop, so each weight costs ~0.5 bytes of memory
traffic instead of 2.

Two kernels, chosen by M (the number of tokens in the call):

* Small M (decode): `_w4a16_splitk`. One row-block of 16 covers all tokens, so
  the grid has only N / BLOCK_N programs; for N = 512 that is far fewer than
  the L4's 58 SMs. Split-K adds a second grid dimension that divides K among
  SPLIT_K programs, each accumulating a partial sum that is atomically added
  into an fp32 output. Memory-bound: the goal is to read the weights at close
  to peak bandwidth.
* Larger M (prefill, batches): `_w4a16_gemm`. A standard tiled matmul over
  (M, N) tiles with the full K loop in each program; becomes compute-bound as M
  grows, where int4 has no bandwidth advantage left to exploit.

Dequantization inside a K-tile: tile rows k map to packed row k // 8 and
nibble k % 8. BLOCK_K divides the group size, so a K-tile lies inside one
group and needs a single row of scales and zeros.

Weight rounding matches the reference path exactly: (q - z) is a small integer,
and q - z times an fp16 scale is correctly rounded to fp16 in both. Results
differ from the reference only by summation order; the split-K path's atomic
adds also make that order nondeterministic run to run (fp32, ~1e-6 relative).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_tile(qw_ptr, s_ptr, z_ptr, offs_k, offs_n, k0, N, G,
                  stride_qk, stride_qn, stride_sg, stride_sn):
    """Load and dequantize the [BLOCK_K, BLOCK_N] weight tile starting at k0."""
    n_mask = offs_n < N
    packed = tl.load(
        qw_ptr + (offs_k[:, None] // 8) * stride_qk + offs_n[None, :] * stride_qn,
        mask=n_mask[None, :], other=0,
    )
    q = (packed >> ((offs_k[:, None] % 8) * 4)) & 0xF
    g = k0 // G
    s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0)
    z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0)
    return (q.to(tl.float16) - z.to(tl.float16)[None, :]) * s[None, :]


@triton.jit
def _w4a16_splitk(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, G,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_span = K // SPLIT_K
    k_start = pid_k * k_span

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_span, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=offs_m[:, None] < M, other=0.0)
        b = _dequant_tile(qw_ptr, s_ptr, z_ptr, offs_k, offs_n, k0, N, G,
                          stride_qk, stride_qn, stride_sg, stride_sn)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w4a16_gemm(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, G,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=offs_m[:, None] < M, other=0.0)
        b = _dequant_tile(qw_ptr, s_ptr, z_ptr, offs_k, offs_n, k0, N, G,
                          stride_qk, stride_qn, stride_sg, stride_sn)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# --- Autotuning -------------------------------------------------------------

def _prune(configs, named_args, **kwargs):
    K, G = named_args["K"], named_args["G"]
    keep = []
    for c in configs:
        bk, sk = c.kwargs["BLOCK_K"], c.kwargs.get("SPLIT_K", 1)
        if G % bk == 0 and K % (bk * sk) == 0:
            keep.append(c)
    return keep


_SPLITK_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=w, num_stages=s)
    for bn in (32, 64, 128)
    for bk in (64, 128)
    for sk in (1, 2, 4, 8)
    for w, s in ((4, 2), (4, 4))
]

_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm, bn, w in ((32, 64, 4), (64, 64, 4), (64, 128, 4), (128, 64, 4), (128, 128, 8))
    for bk in (32, 64)
    for s in (2, 3, 4)
]

_splitk_tuned = triton.autotune(
    configs=_SPLITK_CONFIGS, key=["M", "N", "K"], reset_to_zero=["c_ptr"],
    prune_configs_by={"early_config_prune": _prune},
)(_w4a16_splitk)

_gemm_tuned = triton.autotune(
    configs=_GEMM_CONFIGS, key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune},
)(_w4a16_gemm)


# --- Entry point ------------------------------------------------------------

SMALL_M = 16  # largest M routed to the split-K kernel (BLOCK_M is 16 there)


def _check_config(c: dict, K: int, G: int) -> None:
    bk, sk = c["BLOCK_K"], c.get("SPLIT_K", 1)
    if G % bk or K % (bk * sk):
        raise ValueError(f"BLOCK_K={bk}, SPLIT_K={sk} must tile K={K} and divide group_size={G}")


def w4a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    path: str = "auto",
    config: dict | None = None,
) -> torch.Tensor:
    """x [..., K] fp16 times the packed int4 weight -> [..., N] fp16.

    path: "auto" (split-K for M <= SMALL_M, tiled GEMM above), "splitk" or "gemm".
    config: explicit meta-parameters (BLOCK_M, BLOCK_N, BLOCK_K[, SPLIT_K],
            num_warps, num_stages) that bypass autotuning; used by tests.
    """
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise TypeError("w4a16_matmul expects fp16 activations and scales")
    K = x.shape[-1]
    N = qweight.shape[1]
    if qweight.shape[0] * 8 != K:
        raise ValueError(f"x has K={K} but qweight packs {qweight.shape[0] * 8}")
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    if path == "auto":
        path = "splitk" if M <= SMALL_M else "gemm"
    if path == "splitk" and M > 16:
        raise ValueError("split-K path handles M <= 16")

    args = (
        x2, qweight, scales, qzeros,
    )
    strides = (
        x2.stride(0), x2.stride(1), qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
    )
    if qzeros.stride() != scales.stride():
        raise ValueError("qzeros and scales must share a layout")

    if path == "splitk":
        out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
        if config is None:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])  # noqa: E731
            _splitk_tuned[grid](*args, out, M, N, K, group_size, *strides, out.stride(0), out.stride(1))
        else:
            c = dict(config)
            nw, ns = c.pop("num_warps", 4), c.pop("num_stages", 2)
            _check_config(c, K, group_size)
            grid = (triton.cdiv(N, c["BLOCK_N"]), c["SPLIT_K"])
            _w4a16_splitk[grid](*args, out, M, N, K, group_size, *strides, out.stride(0), out.stride(1),
                                num_warps=nw, num_stages=ns, **c)
        out = out.to(torch.float16)
    elif path == "gemm":
        out = torch.empty(M, N, device=x.device, dtype=torch.float16)
        if config is None:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
            _gemm_tuned[grid](*args, out, M, N, K, group_size, *strides, out.stride(0), out.stride(1))
        else:
            c = dict(config)
            nw, ns = c.pop("num_warps", 4), c.pop("num_stages", 2)
            _check_config(c, K, group_size)
            grid = (triton.cdiv(M, c["BLOCK_M"]), triton.cdiv(N, c["BLOCK_N"]))
            _w4a16_gemm[grid](*args, out, M, N, K, group_size, *strides, out.stride(0), out.stride(1),
                              num_warps=nw, num_stages=ns, **c)
    else:
        raise ValueError(f"unknown path {path!r}")
    return out.reshape(*x.shape[:-1], N)
