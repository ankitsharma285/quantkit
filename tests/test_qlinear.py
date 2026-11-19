import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantkit.model import find_linears, quantize_model_rtn
from quantkit.quant.qlinear import QuantLinear
from quantkit.quant.rtn import dequantize, quantize_rtn


@pytest.mark.parametrize("N,K,G", [(48, 256, 128), (7, 512, 64), (128, 1024, -1)])
@pytest.mark.parametrize("bias", [False, True])
def test_matches_dequantized_reference(N, K, G, bias):
    torch.manual_seed(0)
    w = torch.randn(N, K)
    b = torch.randn(N) if bias else None
    p = quantize_rtn(w, 4, G)
    ql = QuantLinear.from_params(p, b, dtype=torch.float32)

    assert torch.equal(ql.dequantize_weight(torch.float32), dequantize(p, torch.float32))
    x = torch.randn(3, 5, K)
    torch.testing.assert_close(ql(x), F.linear(x, dequantize(p, torch.float32), b))


def test_state_dict_roundtrip():
    p = quantize_rtn(torch.randn(32, 256), 4, 128)
    a = QuantLinear.from_params(p, dtype=torch.float32)
    b = QuantLinear(256, 32, 128, dtype=torch.float32)
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, 256)
    assert torch.equal(a(x), b(x))


def test_storage_is_about_a_quarter_of_fp16():
    ql = QuantLinear(4096, 4096, 128)
    nbytes = sum(t.numel() * t.element_size() for t in ql.buffers())
    fp16 = 4096 * 4096 * 2
    assert nbytes / fp16 < 0.27  # 0.25 for weights + ~1.5% for scales/zeros


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(128, 256), nn.Linear(256, 128))])
        self.lm_head = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(x)


def test_quantize_model_swaps_blocks_but_not_lm_head():
    torch.manual_seed(0)
    m = _Toy()
    ref = {n: l.weight.detach().clone() for n, l in find_linears(m).items()}
    quantize_model_rtn(m, 4, 128, pack=True)
    assert isinstance(m.blocks[0][0], QuantLinear) and isinstance(m.blocks[0][1], QuantLinear)
    assert isinstance(m.lm_head, nn.Linear)
    for name, w in ref.items():
        err = (m.get_submodule(name).dequantize_weight(torch.float32) - w).abs().max()
        assert err < 0.05
    assert m(torch.randn(2, 128)).shape == (2, 10)


def test_fake_mode_keeps_linear_and_allows_3bit():
    m = _Toy()
    quantize_model_rtn(m, bits=3, group_size=128, pack=False)
    assert isinstance(m.blocks[0][0], nn.Linear)
    # Each group of a 3-bit fake-quantized row has at most 8 distinct values.
    row = m.blocks[0][0].weight[0, :128]
    assert row.unique().numel() <= 8


def test_packed_requires_4bit():
    with pytest.raises(ValueError):
        quantize_model_rtn(_Toy(), bits=3, pack=True)
