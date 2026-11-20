"""End-to-end GPTQ on a tiny random Llama (CPU, seconds)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from quantkit.calib.capture import capture_layer0_inputs  # noqa: E402
from quantkit.gptq_model import quantize_model_gptq  # noqa: E402
from quantkit.model import find_linears, quantize_model_rtn  # noqa: E402
from quantkit.quant.qlinear import QuantLinear  # noqa: E402

VOCAB = 256


def _tiny_llama(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg).float().eval()


def _tokens(n, L, seed):
    return torch.randint(0, VOCAB, (n, L), generator=torch.Generator().manual_seed(seed))


def _logit_err(model, ref_logits, x):
    with torch.no_grad():
        return ((model(x).logits - ref_logits) ** 2).mean().item()


def test_capture_returns_one_hidden_state_per_sample():
    m = _tiny_llama()
    inps, kw = capture_layer0_inputs(m, _tokens(3, 32, 0), "cpu")
    assert inps.shape == (3, 32, 256)
    assert "position_embeddings" in kw and kw["use_cache"] is False
    # The real layer 0 must be restored.
    assert type(m.model.layers[0]).__name__ == "LlamaDecoderLayer"


def test_gptq_beats_rtn_on_heldout_logits():
    calib, held = _tokens(16, 64, 1), _tokens(4, 64, 2)
    ref = _tiny_llama()
    with torch.no_grad():
        ref_logits = ref(held).logits

    rtn = _tiny_llama()
    quantize_model_rtn(rtn, bits=3, group_size=64, pack=False)
    gq = _tiny_llama()
    stats = quantize_model_gptq(gq, calib, bits=3, group_size=64, blocksize=64, pack=False,
                                device="cpu", verbose=False)

    assert len(stats) == 2 * 7  # 2 layers x 7 Linears
    e_rtn, e_gptq = _logit_err(rtn, ref_logits, held), _logit_err(gq, ref_logits, held)
    assert e_gptq < e_rtn, (e_gptq, e_rtn)


def test_packed_equals_fake_and_lm_head_untouched():
    calib, held = _tokens(8, 64, 3), _tokens(2, 64, 4)
    fake, packed = _tiny_llama(), _tiny_llama()
    kw = dict(bits=4, group_size=64, blocksize=64, device="cpu", verbose=False)
    quantize_model_gptq(fake, calib, pack=False, **kw)
    quantize_model_gptq(packed, calib, pack=True, **kw)

    assert all(isinstance(m, QuantLinear) for n, m in packed.named_modules() if n.endswith("_proj"))
    assert not find_linears(packed)  # every block Linear replaced
    assert type(packed.lm_head).__name__ == "Linear"
    with torch.no_grad():
        torch.testing.assert_close(packed(held).logits, fake(held).logits, atol=1e-4, rtol=1e-4)


def test_act_order_runs_end_to_end():
    m = _tiny_llama()
    quantize_model_gptq(m, _tokens(4, 64, 5), bits=4, group_size=64, blocksize=64,
                        act_order=True, pack=True, device="cpu", verbose=False)
    with torch.no_grad():
        assert torch.isfinite(m(_tokens(1, 32, 6)).logits).all()
