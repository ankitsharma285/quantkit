"""Applying quantization to a HF causal LM, and saving/loading the result."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .quant.qlinear import QuantLinear
from .quant.rtn import fake_quantize, quantize_rtn

QUANT_CONFIG = "quant_config.json"
QUANT_WEIGHTS = "model.safetensors"


def find_linears(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> dict[str, nn.Linear]:
    """All nn.Linear modules in the transformer blocks (lm_head stays in fp16)."""
    return {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and not any(s in name for s in skip)
    }


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


@torch.no_grad()
def quantize_model_rtn(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    pack: bool = True,
) -> nn.Module:
    """Quantize every block Linear with RTN, in place.

    pack=True  -> replace with QuantLinear (real int4 storage; bits must be 4).
    pack=False -> overwrite weights with fake-quantized fp16 (any bits; for ablations).
    Layers are processed one at a time so peak memory stays near the fp16 model.
    """
    if pack and bits != 4:
        raise ValueError("packed storage is int4 only; pass pack=False for other bit widths")
    for name, lin in find_linears(model).items():
        w = lin.weight.data
        if pack:
            p = quantize_rtn(w, bits, group_size, sym)
            ql = QuantLinear.from_params(p, lin.bias, dtype=w.dtype)
            _set_module(model, name, ql)
        else:
            lin.weight.data = fake_quantize(w, bits, group_size, sym)
    return model


def _untie(model: nn.Module) -> None:
    """Stop HF from re-tying lm_head to the input embedding (Llama-3.2-1B ties them)."""
    model.config.tie_word_embeddings = False
    text_cfg = getattr(model.config, "get_text_config", None)
    if callable(text_cfg):
        text_cfg().tie_word_embeddings = False


@torch.no_grad()
def quantize_lm_head(model: nn.Module, bits: int = 4, group_size: int = 128, sym: bool = False) -> nn.Module:
    """RTN-quantize the output projection, in place, untying it from the input embedding.

    The embedding stays fp16: a decode step reads one row of it, while lm_head
    reads the whole 128256 x 2048 matrix every token. Total stored weights grow
    (the matrix now exists twice: fp16 embedding + int4 lm_head), but bytes read
    per token fall. int4 is packed into QuantLinear; other widths are
    fake-quantized into a separate fp16 matrix (accuracy ablations only).
    """
    lin = model.lm_head
    if not isinstance(lin, nn.Linear):
        raise TypeError(f"lm_head is {type(lin).__name__}, already quantized?")
    _untie(model)
    w = lin.weight.data
    if bits == 4:
        p = quantize_rtn(w, bits, group_size, sym)
        model.lm_head = QuantLinear.from_params(p, lin.bias, dtype=w.dtype)
    else:
        new = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                        device=w.device, dtype=w.dtype)
        new.weight.data.copy_(fake_quantize(w, bits, group_size, sym))
        if lin.bias is not None:
            new.bias.data.copy_(lin.bias.data)
        model.lm_head = new
    return model


def save_quantized(model: nn.Module, out_dir: str | Path, meta: dict) -> None:
    """Save the full state dict (packed int4 layers + fp16 rest) and a config."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Tied weights (Llama-3.2-1B ties lm_head to embed_tokens) share storage, and
    # safetensors refuses shared tensors, so clone them apart.
    state = {k: v.detach().clone().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, out / QUANT_WEIGHTS)
    qlayers = [n for n, m in model.named_modules() if isinstance(m, QuantLinear)]
    (out / QUANT_CONFIG).write_text(json.dumps({**meta, "quantized_layers": qlayers}, indent=2))


def load_quantized(model_id: str, out_dir: str | Path, device: str = "cuda") -> nn.Module:
    """Rebuild the architecture, swap in empty QuantLinears, load packed weights.

    The fp16 model is instantiated first (cheap at 1-3B) so non-persistent buffers
    such as rotary inv_freq are initialised correctly; the meta-device shortcut
    leaves them empty.
    """
    from transformers import AutoModelForCausalLM

    out = Path(out_dir)
    cfg = json.loads((out / QUANT_CONFIG).read_text())
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    lm_head_quantized = "lm_head" in cfg["quantized_layers"]
    if lm_head_quantized:
        _untie(model)
    for name in cfg["quantized_layers"]:
        lin = model.get_submodule(name)
        if name == "lm_head":  # replace the tied module with an independent one before loading
            lin = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
        _set_module(
            model,
            name,
            QuantLinear(
                lin.in_features, lin.out_features, cfg["group_size"],
                bias=lin.bias is not None, dtype=torch.float16, device=device,
            ),
        )
    state = load_file(out / QUANT_WEIGHTS, device=device)
    model.load_state_dict(state, strict=True)
    if not lm_head_quantized:
        model.tie_weights()
    return model.eval()
