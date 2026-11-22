"""Build a model from a config string, shared by the decode and task-evaluation scripts.

    "fp16"                       the original fp16 model
    "w4:<checkpoint dir>"        a packed int4 checkpoint (quantize_eval.py --save)
    options, joined with "+":    "fuse" (q/k/v and gate/up fused),
                                 "lmh"  (lm_head RTN int4 g128, untied from the fp16 embedding)
    e.g. "w4+lmh+fuse:ckpt/gptq-ao-g128", "fp16+lmh"

Quantized modules run on the Triton backend.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from .fuse import fuse_projections
from .model import load_quantized, quantize_lm_head
from .quant.qlinear import set_backend


def load_config(config: str, model_id: str, device: str = "cuda"):
    base, _, path = config.partition(":")
    kind, *opts = base.split("+")
    unknown = set(opts) - {"fuse", "lmh"}
    if unknown:
        raise ValueError(f"unknown options {unknown} in {config!r}")
    if kind == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    elif kind == "w4":
        if not path:
            raise ValueError(f"{config!r}: w4 needs a checkpoint, e.g. w4:ckpt/gptq-ao-g128")
        model = load_quantized(model_id, path, device)
    else:
        raise ValueError(f"bad config {config!r}")
    if "lmh" in opts:
        quantize_lm_head(model, 4, 128)
    set_backend(model, "triton")
    if "fuse" in opts:
        fuse_projections(model)
    return model


def short_name(config: str) -> str:
    """'w4+lmh:/long/path/gptq-ao-g128' -> 'w4+lmh:gptq-ao-g128'."""
    base, sep, path = config.partition(":")
    return base + (sep + path.rstrip("/").split("/")[-1] if path else "")
