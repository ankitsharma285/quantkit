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


def test_lm_head_quantization_unties_and_round_trips(tmp_path):
    from quantkit.model import load_quantized, quantize_lm_head, save_quantized
    from quantkit.quant.qlinear import QuantLinear

    ref = _tiny(5)
    m = _tiny(5)
    emb_before = m.model.embed_tokens.weight.detach().clone()
    quantize_model_rtn(m, 4, 128, pack=True)
    x = _ids(20, 6)
    with torch.no_grad():
        before = m(x).logits  # blocks quantized, lm_head still fp16
    quantize_lm_head(m, 4, 128)
    assert isinstance(m.lm_head, QuantLinear)
    assert torch.equal(m.model.embed_tokens.weight, emb_before)  # embedding untouched, still fp16
    assert m.config.tie_word_embeddings is False

    with torch.no_grad():
        out = m(x).logits
    # lm_head's own int4 error only: small relative to the logits, but not zero.
    rel = ((out - before).norm() / before.norm()).item()
    assert 0 < rel < 0.2, rel

    # Save and reload: lm_head must come back quantized and *not* re-tied to the embedding.
    ref.save_pretrained(tmp_path / "base")
    save_quantized(m, tmp_path / "q", {"method": "rtn", "bits": 4, "group_size": 128})
    r = load_quantized(str(tmp_path / "base"), tmp_path / "q", device="cpu").float()
    assert isinstance(r.lm_head, QuantLinear)
    with torch.no_grad():
        torch.testing.assert_close(r(x).logits, out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(r.model.embed_tokens.weight.float(), emb_before.half().float())

    # Works together with fusion.
    fuse_projections(r)
    with torch.no_grad():
        torch.testing.assert_close(r(x).logits, out, atol=2e-2, rtol=2e-2)


def test_lm_head_fake_quant_other_bits_leaves_embedding_alone():
    from quantkit.model import quantize_lm_head

    m = _tiny(7)
    emb = m.model.embed_tokens.weight.detach().clone()
    quantize_lm_head(m, bits=8, group_size=128)
    assert type(m.lm_head).__name__ == "Linear"
    assert m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()
    assert torch.equal(m.model.embed_tokens.weight, emb)
