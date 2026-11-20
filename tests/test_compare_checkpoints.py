"""The GPTQ-format decoder in scripts/compare_checkpoints.py must read back exactly what we encode."""
import importlib.util
from pathlib import Path

import pytest
import torch

from quantkit.quant.pack import pack_int4
from quantkit.quant.qlinear import QuantLinear
from quantkit.quant.rtn import dequantize, quantize_rtn

_spec = importlib.util.spec_from_file_location(
    "compare_checkpoints", Path(__file__).parents[1] / "scripts" / "compare_checkpoints.py"
)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _pack_along_n(z: torch.Tensor) -> torch.Tensor:
    """[R, N] values 0..15 -> [R, N/8] int32, nibbles along N, little first."""
    return pack_int4(z.t().contiguous()).t().contiguous()


@pytest.mark.parametrize("v1_offset", [True, False])
def test_decodes_gptq_v1_and_v2_zero_points(v1_offset):
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    p = quantize_rtn(W, 4, 128)
    ql = QuantLinear.from_params(p, dtype=torch.float16)
    prefix = "model.layers.0.self_attn.q_proj"
    ours_sd = {prefix + k: v for k, v in
               {".qweight": ql.qweight, ".scales": ql.scales, ".qzeros": ql.qzeros}.items()}

    zeros = ql.qzeros.int()
    stored = (zeros - 1) & 0xF if v1_offset else zeros
    theirs_sd = {
        prefix + ".qweight": ql.qweight.clone(),
        prefix + ".qzeros": _pack_along_n(stored),
        prefix + ".scales": ql.scales.clone(),
        prefix + ".g_idx": (torch.arange(256) // 128).int(),
    }
    qa, za, sa, ga = cc.ours(ours_sd, prefix)
    qb, zb, sb, gb, off = cc.theirs(theirs_sd, prefix, W)
    assert off == (1 if v1_offset else 0)
    Wa, Wb = cc.dequant(qa, za, sa, ga), cc.dequant(qb, zb, sb, gb)
    assert torch.equal(Wa, Wb)
    torch.testing.assert_close(Wa, dequantize(p, torch.float32), atol=2e-3, rtol=1e-3)
