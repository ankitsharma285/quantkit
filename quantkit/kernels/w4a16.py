"""Fused W4A16 matmul in Triton: out[M, N] = x[M, K] @ dequant(W)[K, N].

Weights stay packed in global memory (see quant/pack.py) and are dequantized in
registers inside the matmul loop, so each weight costs ~0.5 bytes of memory
traffic instead of 2.

Three kernels:

* `_w4a16_gemm`: tiled matmul over (M, N) tiles, full K loop per program, fp16
  output written directly. BLOCK_M can be 16, so it also covers decode with a
  single launch.
* `_w4a16_splitk` (M <= 16): adds a grid dimension that divides K among
  SPLIT_K programs, which atomically add fp32 partial sums. More programs for
  narrow layers (k/v_proj has N = 512), at the cost of two extra launches
  (zero-fill and fp16 cast).
* `_w4a16_gemv` (M == 1): no tensor cores; multiply-and-sum in fp32 registers,
  avoiding the 16-row padding tl.dot needs. Split-K with atomics as above.

Which one is fastest depends on the shape, so `path="auto"` measures every
candidate kernel in both unpack variants on first use for each (M, N, K) and
caches the winner (GPU only; the CPU interpreter falls back to a fixed rule).
On the L4 neither variant wins everywhere: unpack-once is up to 25% faster on
large layers, per-nibble up to 25% faster on k/v_proj and at M = 32.

Dequantization of a [BLOCK_K, BLOCK_N] tile, two variants (UNPACK_ONCE):
* per-nibble (Phase 4): load packed row k // 8 once per k, i.e. each int32 is
  requested 8 times and served from cache, then shift by 4 * (k % 8).
* once (Phase 5): load the [BLOCK_K / 8, BLOCK_N] packed rows once, expand to
  [BLOCK_K / 8, 8, BLOCK_N] with 8 shifts, reshape to [BLOCK_K, BLOCK_N].
BLOCK_K divides the group size, so a tile needs one row of scales and zeros.

Weight rounding matches the reference path exactly: (q - z) is a small integer,
and (q - z) times an fp16 scale is correctly rounded to fp16 in both. Results
differ from the reference only by summation order; the atomic paths also make
that order nondeterministic run to run (fp32 partials, ~1e-6 relative).
"""
from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _dequant_tile(qw_ptr, s_ptr, z_ptr, k0, offs_n, N, G,
                  stride_qk, stride_qn, stride_sg, stride_sn,
                  BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, UNPACK_ONCE: tl.constexpr):
    """Load and dequantize the [BLOCK_K, BLOCK_N] fp16 weight tile starting at row k0."""
    n_mask = offs_n < N
    if UNPACK_ONCE:
        offs_r = k0 // 8 + tl.arange(0, BLOCK_K // 8)
        packed = tl.load(qw_ptr + offs_r[:, None] * stride_qk + offs_n[None, :] * stride_qn,
                         mask=n_mask[None, :], other=0)
        shifts = tl.arange(0, 8) * 4
        q = (packed[:, None, :] >> shifts[None, :, None]) & 0xF
        q = tl.reshape(q, (BLOCK_K, BLOCK_N))
    else:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        packed = tl.load(qw_ptr + (offs_k[:, None] // 8) * stride_qk + offs_n[None, :] * stride_qn,
                         mask=n_mask[None, :], other=0)
        q = (packed >> ((offs_k[:, None] % 8) * 4)) & 0xF
    g = k0 // G
    s = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0.0)
    z = tl.load(z_ptr + g * stride_sg + offs_n * stride_sn, mask=n_mask, other=0)
    return (q.to(tl.float16) - z.to(tl.float16)[None, :]) * s[None, :]


@triton.jit
def _w4a16_gemm(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, G,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, UNPACK_ONCE: tl.constexpr,
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
        b = _dequant_tile(qw_ptr, s_ptr, z_ptr, k0, offs_n, N, G, stride_qk, stride_qn,
                          stride_sg, stride_sn, BLOCK_K, BLOCK_N, UNPACK_ONCE)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w4a16_splitk(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, G,
    stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    UNPACK_ONCE: tl.constexpr,
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
        b = _dequant_tile(qw_ptr, s_ptr, z_ptr, k0, offs_n, N, G, stride_qk, stride_qn,
                          stride_sg, stride_sn, BLOCK_K, BLOCK_N, UNPACK_ONCE)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _w4a16_gemv(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    N, K, G,
    stride_ak, stride_qk, stride_qn, stride_sg, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, UNPACK_ONCE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_span = K // SPLIT_K
    k_start = pid_k * k_span

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_span, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_k * stride_ak).to(tl.float32)
        b = _dequant_tile(qw_ptr, s_ptr, z_ptr, k0, offs_n, N, G, stride_qk, stride_qn,
                          stride_sg, stride_sn, BLOCK_K, BLOCK_N, UNPACK_ONCE)
        acc += tl.sum(a[:, None] * b.to(tl.float32), axis=0)

    tl.atomic_add(c_ptr + offs_n, acc, mask=offs_n < N)


# --- Autotuning -------------------------------------------------------------

def _prune(configs, named_args, **kwargs):
    K, G = named_args["K"], named_args["G"]
    return [c for c in configs
            if G % c.kwargs["BLOCK_K"] == 0 and K % (c.kwargs["BLOCK_K"] * c.kwargs.get("SPLIT_K", 1)) == 0]


# Short timing per candidate config, so several kernel variants can be tuned in one session.
_tune_bench = functools.partial(do_bench, warmup=5, rep=25)

_GEMM_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm, bn, w in ((16, 32, 4), (16, 64, 4), (16, 128, 4), (32, 64, 4), (64, 64, 4),
                      (64, 128, 4), (128, 64, 4), (128, 128, 8))
    for bk in (32, 64, 128)
    for s in (2, 4)
]
_SPLITK_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=4, num_stages=s)
    for bn in (32, 64, 128)
    for bk in (64, 128)
    for sk in (2, 4, 8)
    for s in (2, 4)
]
_GEMV_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=w, num_stages=2)
    for bn in (16, 32, 64, 128)
    for bk in (64, 128)
    for sk in (1, 2, 4, 8)
    for w in (2, 4)
]

_tune = functools.partial(triton.autotune, prune_configs_by={"early_config_prune": _prune},
                          do_bench=_tune_bench)
_gemm_tuned = _tune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "UNPACK_ONCE"])(_w4a16_gemm)
_splitk_tuned = _tune(configs=_SPLITK_CONFIGS, key=["M", "N", "K", "UNPACK_ONCE"],
                      reset_to_zero=["c_ptr"])(_w4a16_splitk)
_gemv_tuned = _tune(configs=_GEMV_CONFIGS, key=["N", "K", "UNPACK_ONCE"],
                    reset_to_zero=["c_ptr"])(_w4a16_gemv)


# --- Entry point ------------------------------------------------------------

SMALL_M = 16  # largest M the split-K kernel accepts (BLOCK_M is 16 there)
PATHS = ("gemm", "splitk", "gemv")
_dispatch_cache: dict[tuple, tuple[str, bool]] = {}


def _check_config(c: dict, K: int, G: int) -> None:
    bk, sk = c["BLOCK_K"], c.get("SPLIT_K", 1)
    if G % bk or K % (bk * sk):
        raise ValueError(f"BLOCK_K={bk}, SPLIT_K={sk} must tile K={K} and divide group_size={G}")


def candidate_paths(M: int) -> list[str]:
    return ["gemm"] + (["splitk"] if M <= SMALL_M else []) + (["gemv"] if M == 1 else [])


def _launch(path, x2, qweight, scales, qzeros, G, unpack_once, config):
    M, K = x2.shape
    N = qweight.shape[1]
    common = (x2, qweight, scales, qzeros)
    s_strides = (qweight.stride(0), qweight.stride(1), scales.stride(0), scales.stride(1))
    if config is not None:
        config = dict(config)
        nw, ns = config.pop("num_warps", 4), config.pop("num_stages", 2)
        _check_config(config, K, G)

    if path == "gemm":
        out = torch.empty(M, N, device=x2.device, dtype=torch.float16)
        args = (*common, out, M, N, K, G, x2.stride(0), x2.stride(1), *s_strides,
                out.stride(0), out.stride(1))
        if config is None:
            grid = lambda m: (triton.cdiv(M, m["BLOCK_M"]), triton.cdiv(N, m["BLOCK_N"]))  # noqa: E731
            _gemm_tuned[grid](*args, UNPACK_ONCE=unpack_once)
        else:
            grid = (triton.cdiv(M, config["BLOCK_M"]), triton.cdiv(N, config["BLOCK_N"]))
            _w4a16_gemm[grid](*args, UNPACK_ONCE=unpack_once, num_warps=nw, num_stages=ns, **config)
        return out

    out = torch.zeros(M, N, device=x2.device, dtype=torch.float32)
    if path == "splitk":
        if M > SMALL_M:
            raise ValueError(f"split-K path handles M <= {SMALL_M}")
        args = (*common, out, M, N, K, G, x2.stride(0), x2.stride(1), *s_strides,
                out.stride(0), out.stride(1))
        if config is None:
            grid = lambda m: (triton.cdiv(N, m["BLOCK_N"]), m["SPLIT_K"])  # noqa: E731
            _splitk_tuned[grid](*args, UNPACK_ONCE=unpack_once)
        else:
            grid = (triton.cdiv(N, config["BLOCK_N"]), config["SPLIT_K"])
            _w4a16_splitk[grid](*args, UNPACK_ONCE=unpack_once, num_warps=nw, num_stages=ns, **config)
    elif path == "gemv":
        if M != 1:
            raise ValueError("gemv path handles M == 1")
        args = (*common, out, N, K, G, x2.stride(1), *s_strides)
        if config is None:
            grid = lambda m: (triton.cdiv(N, m["BLOCK_N"]), m["SPLIT_K"])  # noqa: E731
            _gemv_tuned[grid](*args, UNPACK_ONCE=unpack_once)
        else:
            grid = (triton.cdiv(N, config["BLOCK_N"]), config["SPLIT_K"])
            _w4a16_gemv[grid](*args, UNPACK_ONCE=unpack_once, num_warps=nw, num_stages=ns, **config)
    else:
        raise ValueError(f"unknown path {path!r}")
    return out.to(torch.float16)


def _pick(x2, qweight, scales, qzeros, G, unpack_once) -> tuple[str, bool]:
    """(path, unpack_once) for this shape. unpack_once=None lets the measurement choose it."""
    M, K = x2.shape
    N = qweight.shape[1]
    measured = x2.is_cuda and os.environ.get("TRITON_INTERPRET") != "1"
    if not measured:
        return ("splitk" if M <= SMALL_M else "gemm"), (True if unpack_once is None else unpack_once)
    key = (M, N, K, G, unpack_once, x2.device.index)
    if key not in _dispatch_cache:
        variants = (False, True) if unpack_once is None else (unpack_once,)
        times = {}
        for p in candidate_paths(M):
            for u in variants:
                fn = functools.partial(_launch, p, x2, qweight, scales, qzeros, G, u, None)
                fn()  # autotune this kernel for this shape
                times[(p, u)] = do_bench(fn, warmup=5, rep=25, return_mode="median")
        _dispatch_cache[key] = min(times, key=times.get)
    return _dispatch_cache[key]


def dispatch_choice(M: int, N: int, K: int, G: int = 128, unpack_once: bool | None = None):
    """What path="auto" chose for this shape, or None if it hasn't run yet."""
    return next((v for k, v in _dispatch_cache.items() if k[:5] == (M, N, K, G, unpack_once)), None)


def w4a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    path: str = "auto",
    config: dict | None = None,
    unpack_once: bool | None = None,
) -> torch.Tensor:
    """x [..., K] fp16 times the packed int4 weight -> [..., N] fp16.

    path: "auto" (fastest kernel for this shape, measured once and cached;
          fixed rule without a GPU), "gemm", "splitk" or "gemv".
    config: explicit meta-parameters (BLOCK_*, SPLIT_K, num_warps, num_stages)
            that bypass autotuning; used by tests.
    unpack_once: load each packed int32 once per tile (Phase 5) or once per
            nibble (Phase 4). None: chosen by measurement with path="auto",
            True with an explicit path.
    """
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise TypeError("w4a16_matmul expects fp16 activations and scales")
    if qzeros.stride() != scales.stride():
        raise ValueError("qzeros and scales must share a layout")
    K = x.shape[-1]
    N = qweight.shape[1]
    if qweight.shape[0] * 8 != K:
        raise ValueError(f"x has K={K} but qweight packs {qweight.shape[0] * 8}")
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if path == "auto":
        path, unpack_once = _pick(x2, qweight, scales, qzeros, group_size, unpack_once)
    elif unpack_once is None:
        unpack_once = True
    out = _launch(path, x2, qweight, scales, qzeros, group_size, unpack_once, config)
    return out.reshape(*x.shape[:-1], N)
