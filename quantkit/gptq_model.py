"""Apply GPTQ to a HF decoder-only LM, one decoder layer at a time.

For each layer:
  1. Run the calibration hidden states through the layer, with hooks on the
     Linears of the current sub-group, accumulating each Linear's Hessian.
  2. GPTQ-quantize that sub-group and write the dequantized weights back, so
     the next sub-group sees the error already made ("true sequential").
     Order: q/k/v -> o -> gate/up -> down.
  3. Run the layer once more (now fully quantized) to produce the next
     layer's inputs, so later layers calibrate on what the quantized model
     actually computes.

Only one layer's activations (inputs and outputs, ~1 GB each for 128 x 2048
tokens on Llama-3.2-1B in fp16) and one layer's Hessians live on the GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .calib.capture import capture_layer0_inputs, decoder_layers
from .model import _set_module
from .quant.gptq import HessianAccumulator, gptq_quantize
from .quant.qlinear import QuantLinear
from .quant.rtn import dequantize

# Sub-groups in execution order for Llama-style blocks.
SEQUENTIAL = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def _layer_groups(layer: nn.Module, true_sequential: bool) -> list[list[str]]:
    linears = {n for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
    if not true_sequential:
        return [sorted(linears)]
    groups = [[n for n in g if n in linears] for g in SEQUENTIAL]
    groups = [g for g in groups if g]
    covered = {n for g in groups for n in g}
    if covered != linears:
        raise ValueError(f"unrecognized Linears {sorted(linears - covered)}; pass true_sequential=False")
    return groups


def _run_layer(layer: nn.Module, x: torch.Tensor, kwargs: dict) -> torch.Tensor:
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def quantize_model_gptq(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
    true_sequential: bool = True,
    pack: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Quantize every decoder-layer Linear in place. Returns per-layer stats.

    pack=True swaps in QuantLinear (int4 only); pack=False keeps nn.Linear with
    dequantized weights (any bit width; for ablations).
    """
    if pack and bits != 4:
        raise ValueError("packed storage is int4 only; pass pack=False for other bit widths")
    model.eval()
    dtype = next(model.parameters()).dtype
    inps, kwargs = capture_layer0_inputs(model, calib_tokens, device)
    outs = torch.empty_like(inps)
    n = inps.shape[0]
    layer_stats: dict[str, dict] = {}

    for li, layer in enumerate(decoder_layers(model)):
        t0 = time.perf_counter()
        params = {}
        for group in _layer_groups(layer, true_sequential):
            mods = {name: layer.get_submodule(name) for name in group}
            accs = {name: HessianAccumulator(m.in_features, device=device) for name, m in mods.items()}
            hooks = [
                m.register_forward_hook(lambda mod, inp, out, a=accs[name]: a.add_batch(inp[0]))
                for name, m in mods.items()
            ]
            for j in range(n):
                _run_layer(layer, inps[j : j + 1], kwargs)
            for h in hooks:
                h.remove()

            for name, m in mods.items():
                p, st = gptq_quantize(
                    m.weight.data, accs[name].H, bits=bits, group_size=group_size, sym=sym,
                    blocksize=blocksize, percdamp=percdamp, act_order=act_order,
                )
                m.weight.data = dequantize(p, dtype)
                params[name] = p
                layer_stats[f"layers.{li}.{name}"] = st
            del accs

        for j in range(n):
            outs[j : j + 1] = _run_layer(layer, inps[j : j + 1], kwargs)

        if pack:
            for name, p in params.items():
                lin = layer.get_submodule(name)
                _set_module(layer, name, QuantLinear.from_params(p, lin.bias, dtype=dtype))

        inps, outs = outs, inps
        if verbose:
            worst = max(params, key=lambda k: layer_stats[f"layers.{li}.{k}"]["loss"])
            print(
                f"layer {li:2d}: {time.perf_counter() - t0:5.1f}s  "
                f"max loss {layer_stats[f'layers.{li}.{worst}']['loss']:.3g} ({worst})",
                flush=True,
            )

    del inps, outs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return layer_stats
