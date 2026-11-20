import pytest
import torch

from quantkit.quant.gptq import HessianAccumulator, _scale_zero, gptq_quantize
from quantkit.quant.rtn import dequantize, quantize_rtn


def _problem(N=32, K=256, T=2048, seed=0, dtype=torch.float64):
    """Weights plus *correlated* inputs with a few outlier channels, like real activations."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(N, K, generator=g, dtype=dtype)
    mix = torch.randn(K, K, generator=g, dtype=dtype) / K**0.5
    X = torch.randn(T, K, generator=g, dtype=dtype) @ (torch.eye(K, dtype=dtype) + mix)
    X[:, :4] *= 10.0  # outlier input channels
    acc = HessianAccumulator(K)
    acc.H = acc.H.to(dtype)
    acc.add_batch(X)
    return W, acc.H, X


def _naive_obs(W, H, bits, G, sym, percdamp):
    """Textbook OBS: one column at a time, explicit H^-1 with the column removed after each step.

    No Cholesky, no blocking; shares only the scale/zero rule with the real implementation.
    """
    W, H = W.clone(), H.clone()
    N, K = W.shape
    qmax = 2**bits - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.inv(H)
    Q = torch.zeros_like(W)
    for j in range(K):
        if j % G == 0:
            s, z = _scale_zero(W[:, j : j + G], bits, sym)
            s, z = s.squeeze(1).to(W.dtype), z.squeeze(1).to(W.dtype)
        w = W[:, j]
        q = (torch.clamp(torch.round(w / s) + z, 0, qmax) - z) * s
        Q[:, j] = q
        err = (w - q) / Hinv[j, j]
        W[:, j:] -= err.unsqueeze(1) * Hinv[j, j:].unsqueeze(0)
        Hinv = Hinv - Hinv[:, j : j + 1] @ Hinv[j : j + 1, :] / Hinv[j, j]
    return Q


def _out_err(W, Wq, X):
    return ((X @ (W - Wq).t()) ** 2).mean().item()


@pytest.mark.parametrize("G,blocksize", [(128, 128), (32, 128), (256, 128), (-1, 64)])
def test_blocked_cholesky_matches_naive_obs(G, blocksize):
    W, H, _ = _problem()
    p, _ = gptq_quantize(W, H, bits=4, group_size=G, blocksize=blocksize)
    Q = dequantize(p, torch.float64)
    Qref = _naive_obs(W, H, 4, W.shape[1] if G == -1 else G, False, 0.01)
    # Same algorithm, different arithmetic order: allow a handful of flipped roundings.
    agree = (Q - Qref).abs() < 1e-6
    assert agree.float().mean() > 0.995
    torch.testing.assert_close(Q, Qref, atol=0.3, rtol=0)


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("act_order", [False, True])
def test_beats_rtn_on_output_error(bits, act_order):
    W, H, X = _problem(seed=1)
    p, _ = gptq_quantize(W, H, bits=bits, group_size=128, act_order=act_order)
    e_gptq = _out_err(W, dequantize(p, torch.float64), X)
    e_rtn = _out_err(W, dequantize(quantize_rtn(W, bits, 128), torch.float64), X)
    assert e_gptq < 0.7 * e_rtn, (e_gptq, e_rtn)


def test_params_are_self_consistent_and_storable():
    W, H, _ = _problem(seed=2)
    p, stats = gptq_quantize(W.float(), H.float(), bits=4, group_size=64)
    assert p.q.dtype == torch.uint8 and p.q.max() <= 15 and p.zeros.max() <= 15
    assert p.scales.shape == (W.shape[0], W.shape[1] // 64)
    # Scales are already fp16-representable, so storing them in fp16 changes nothing.
    assert torch.equal(p.scales, p.scales.half().float())
    assert stats["loss"] > 0 and stats["dead_cols"] == 0


def test_act_order_keeps_groups_contiguous():
    """Static groups: every group's scale/zero comes from the original contiguous columns."""
    W, H, _ = _problem(seed=3, dtype=torch.float32)
    p, _ = gptq_quantize(W, H, bits=4, group_size=64, act_order=True)
    for g in range(W.shape[1] // 64):
        s, z = _scale_zero(W[:, g * 64 : (g + 1) * 64], 4, False)
        torch.testing.assert_close(p.scales[:, g], s.squeeze(1))
        assert torch.equal(p.zeros[:, g], z.squeeze(1).to(torch.uint8))


def test_dead_input_channels():
    W, H, X = _problem(seed=4, dtype=torch.float32)
    H[5, :] = 0
    H[:, 5] = 0
    p, stats = gptq_quantize(W, H, bits=4, group_size=128)
    assert stats["dead_cols"] == 1
    assert torch.isfinite(dequantize(p, torch.float32)).all()


def test_hessian_accumulator_matches_direct_formula():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(3, 50, 16, generator=g)  # batch, tokens, features
    acc = HessianAccumulator(16)
    for b in range(3):
        acc.add_batch(X[b])
    flat = X.reshape(-1, 16)
    torch.testing.assert_close(acc.H, 2 * flat.t() @ flat / flat.shape[0], rtol=1e-5, atol=1e-5)


def test_rejects_incompatible_group_and_block():
    W, H, _ = _problem(K=384)
    with pytest.raises(ValueError):
        gptq_quantize(W, H, group_size=96, blocksize=128)
