import pytest
import torch

from quantkit.quant.rtn import dequantize, fake_quantize, quantize_rtn


def _w(N=64, K=256, seed=0):
    return torch.randn(N, K, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [32, 128, -1])
@pytest.mark.parametrize("sym", [False, True])
def test_error_bounded_by_half_step(bits, group_size, sym):
    w = _w()
    p = quantize_rtn(w, bits, group_size, sym)
    G = p.group_size
    err = (dequantize(p, torch.float32) - w).abs().view(w.shape[0], -1, G)
    half_step = p.scales.unsqueeze(-1) / 2
    if sym:
        # The symmetric grid is [-8s, 7s] (4-bit); +absmax clips by up to one step.
        assert (err <= 2 * half_step + 1e-5).all()
    else:
        assert (err <= half_step + 1e-5).all()


def test_codes_in_range_and_zero_representable():
    p = quantize_rtn(_w(), bits=4, group_size=128)
    assert p.q.max() <= 15 and p.zeros.max() <= 15
    # Asymmetric with 0 forced into range: a zero weight dequantizes to ~0.
    w = _w()
    w[:, :128] = 0.0
    deq = dequantize(quantize_rtn(w, 4, 128), torch.float32)
    assert deq[:, :128].abs().max() < 1e-6


def test_smaller_groups_reduce_error():
    w = _w(256, 1024)
    errs = [(fake_quantize(w, 4, g) - w).pow(2).mean().item() for g in (-1, 256, 64)]
    assert errs[0] > errs[1] > errs[2]


def test_more_bits_reduce_error():
    w = _w(256, 1024)
    errs = [(fake_quantize(w, b, 128) - w).pow(2).mean().item() for b in (2, 3, 4, 8)]
    assert errs == sorted(errs, reverse=True)


def test_constant_group_does_not_nan():
    w = torch.full((4, 128), 0.5)
    assert torch.isfinite(fake_quantize(w, 4, 128)).all()


def test_rejects_indivisible_group():
    with pytest.raises(ValueError):
        quantize_rtn(_w(K=200), 4, 128)
