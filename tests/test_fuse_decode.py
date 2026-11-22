"""Fusion must not change outputs; the static-cache decoder must match HF generate."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from quantkit.decode import Decoder  # noqa: E402
from quantkit.fuse import fuse_projections  # noqa: E402
from quantkit.model import quantize_model_rtn  # noqa: E402

VOCAB = 256


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                      max_position_embeddings=512)
    return LlamaForCausalLM(cfg).float().eval()


def _ids(L, seed):
    return torch.randint(0, VOCAB, (1, L), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("quantized", [False, True])
def test_fusion_preserves_logits(quantized):
    a, b = _tiny(), _tiny()
    if quantized:
        quantize_model_rtn(a, 4, 128, pack=True)
        quantize_model_rtn(b, 4, 128, pack=True)
    x = _ids(24, 1)
    with torch.no_grad():
        ref = a(x).logits
    assert fuse_projections(b) == 4  # 2 layers x (qkv, gate_up)
    with torch.no_grad():
        out = b(x).logits
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    # The fused group owns the weights once: no duplicated parameters or buffers.
    names = [n for n, _ in b.named_buffers()] + [n for n, _ in b.named_parameters()]
    assert not any(n.split(".")[-2] in ("q_proj", "k_proj", "gate_proj") for n in names)


def test_slice_recomputes_on_unexpected_input():
    m = _tiny()
    fuse_projections(m, gate_up=False)
    attn = m.model.layers[0].self_attn
    x1, x2 = torch.randn(1, 3, 256), torch.randn(1, 3, 256)
    attn.q_proj(x1)
    k_from_x2 = attn.k_proj(x2)  # different tensor than the one q_proj saw
    full = attn.qkv_fused.fused(x2)
    torch.testing.assert_close(k_from_x2, full[..., 256:384])


@pytest.mark.parametrize("fuse", [False, True])
def test_static_decoder_matches_hf_generate(fuse):
    m = _tiny(3)
    prompt = _ids(16, 4)
    with torch.no_grad():
        ref = m.generate(prompt, max_new_tokens=12, do_sample=False, min_new_tokens=12)[:, 16:]
    if fuse:
        fuse_projections(m)
    d = Decoder(m, prompt_len=16, new_tokens=12)
    out = d.generate(prompt)
    assert torch.equal(out, ref)
    assert torch.equal(d.generate(prompt), out)  # cache reset between runs
