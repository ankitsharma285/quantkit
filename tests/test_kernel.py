"""W4A16 Triton kernel vs the reference dequantize-then-matmul path.

On CPU these run in Triton's interpreter with explicit configs (small shapes);
on a GPU the `gpu` tests also exercise autotuning on the real Llama-3.2-1B
layer shapes.
"""
import os

import pytest
import torch

triton = pytest.importorskip("triton")

from quantkit.kernels.w4a16 import w4a16_matmul  # noqa: E402
from quantkit.quant.qlinear import QuantLinear, set_backend  # noqa: E402
from quantkit.quant.rtn import quantize_rtn  # noqa: E402

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
INTERP = os.environ.get("TRITON_INTERPRET") == "1"
needs_triton = pytest.mark.skipif(not (CUDA or INTERP), reason="needs a GPU or TRITON_INTERPRET=1")


def _layer(N, K, G, seed=0, bias=False):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(N, K, generator=g) * 0.05
    b = torch.randn(N, generator=g) if bias else None
    return QuantLinear.from_params(quantize_rtn(W, 4, G), b, dtype=torch.float16).to(DEV)


def _ref(ql, x):
    # fp32 reference on the exact fp16 dequantized weights the kernel computes.
    return torch.nn.functional.linear(x.float(), ql.dequantize_weight(torch.float16).float())


def _check(out, ref):
    assert out.dtype == torch.float16 and out.shape == ref.shape
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 2e-3 * scale + 1e-3, (err, scale)


SPLITK = [
    dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, SPLIT_K=1),
    dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, SPLIT_K=4),
    dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, SPLIT_K=2),
]
GEMM = [
    dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32),
    dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64),
]


@needs_triton
@pytest.mark.parametrize("cfg", SPLITK)
@pytest.mark.parametrize("M", [1, 3, 16])
@pytest.mark.parametrize("N", [64, 72])  # 72: N not a multiple of BLOCK_N
def test_splitk_matches_reference(cfg, M, N):
    K, G = 256, 128
    ql = _layer(N, K, G, seed=M + N)
    x = torch.randn(M, K, device=DEV).half()
    out = w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, G, path="splitk", config=cfg)
    _check(out, _ref(ql, x))


@needs_triton
@pytest.mark.parametrize("cfg", GEMM)
@pytest.mark.parametrize("M", [17, 40])  # not multiples of BLOCK_M
def test_gemm_matches_reference(cfg, M):
    N, K, G = 96, 256, 64
    ql = _layer(N, K, G, seed=M)
    x = torch.randn(M, K, device=DEV).half()
    out = w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, G, path="gemm", config=cfg)
    _check(out, _ref(ql, x))


@needs_triton
def test_extreme_codes_and_zero_points():
    """All-15 codes with zero point 0 and all-0 codes with zero point 15 (sign-bit nibble)."""
    N, K, G = 32, 128, 128
    ql = _layer(N, K, G)
    ql.qweight.fill_(-1)  # every nibble = 15
    ql.qzeros.zero_()
    x = torch.ones(2, K, device=DEV).half()
    out = w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, G, path="splitk", config=SPLITK[0])
    _check(out, _ref(ql, x))
    ql.qweight.zero_()
    ql.qzeros.fill_(15)
    out = w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, G, path="gemm", config=GEMM[0])
    _check(out, _ref(ql, x))


@needs_triton
def test_batched_leading_dims_and_bias_via_quantlinear():
    if INTERP and not CUDA:
        pytest.skip("autotuned path needs a GPU")
    ql = _layer(128, 256, 128, bias=True)
    x = torch.randn(2, 5, 256, device=DEV).half()
    ref = ql(x)
    set_backend(ql, "triton")
    torch.testing.assert_close(ql(x), ref, atol=2e-2, rtol=2e-2)


def test_rejects_bad_inputs():
    ql = _layer(32, 128, 128)
    with pytest.raises(TypeError):
        w4a16_matmul(torch.randn(1, 128, device=DEV), ql.qweight, ql.scales, ql.qzeros, 128)
    with pytest.raises(ValueError):
        w4a16_matmul(torch.randn(1, 64, device=DEV).half(), ql.qweight, ql.scales, ql.qzeros, 128)
    with pytest.raises(ValueError):
        w4a16_matmul(torch.randn(1, 128, device=DEV).half(), ql.qweight, ql.scales, ql.qzeros, 128,
                     path="splitk", config=dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, SPLIT_K=1))


# --- GPU: real layer shapes, autotuned -----------------------------------------------

LLAMA_1B = [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)]  # (N, K)


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("N,K", LLAMA_1B)
@pytest.mark.parametrize("M", [1, 2, 7, 16, 17, 64, 255])
def test_autotuned_llama_shapes(N, K, M):
    ql = _layer(N, K, 128, seed=N + K + M)
    x = torch.randn(M, K, device=DEV).half()
    out = w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, 128)
    _check(out, _ref(ql, x))
