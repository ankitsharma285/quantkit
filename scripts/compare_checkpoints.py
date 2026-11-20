"""Compare a quantkit checkpoint with a GPTQModel checkpoint, module by module.

    python scripts/compare_checkpoints.py ckpt/gptq-ao ckpt/ref-gptqmodel-ao
    python scripts/compare_checkpoints.py ckpt/gptq-ao ckpt/gptq-ao-tf32     # two quantkit runs

Both quantize the same fp16 model with the same calibration windows. Walking the
modules in execution order shows where the two implementations start to differ:
- if layer 0's first modules already disagree, the per-matrix algorithm differs
  (same weights, same inputs);
- if early modules agree and later ones drift, the difference is in how
  calibration inputs are propagated between modules/layers.

GPTQ checkpoint layout (GPTQModel FORMAT.GPTQ, a.k.a. "v1"):
    qweight [K/8, N] int32  nibbles along K, little first (same as quantkit)
    qzeros  [K/G, N/8] int32 nibbles along N, little first, stored as zero - 1
    scales  [K/G, N] fp16
    g_idx   [K] int32 group of each input channel
The "- 1" is detected, not assumed: both conventions are tried and the one whose
dequantized weights are closer to the original fp16 weights is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from quantkit.quant.pack import unpack_int4


def load_dir(d: str | Path) -> dict[str, torch.Tensor]:
    files = sorted(Path(d).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {d}")
    out: dict[str, torch.Tensor] = {}
    for f in files:
        out.update(load_file(f))
    return out


def unpack_along_n(packed: torch.Tensor) -> torch.Tensor:
    """[R, N/8] int32 with nibbles along the last dim -> [R, N] uint8."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF
    return q.reshape(packed.shape[0], -1).to(torch.uint8)


def dequant(q: torch.Tensor, z: torch.Tensor, s: torch.Tensor, g_idx: torch.Tensor) -> torch.Tensor:
    """q [K, N], z/s [n_groups, N], g_idx [K] -> W [N, K] fp32."""
    return ((q.float() - z[g_idx].float()) * s[g_idx].float()).t()


def ours(sd: dict, prefix: str):
    q = unpack_int4(sd[prefix + ".qweight"])
    z, s = sd[prefix + ".qzeros"], sd[prefix + ".scales"]
    K = q.shape[0]
    g_idx = torch.arange(K) // (K // s.shape[0])
    return q, z, s, g_idx


def theirs(sd: dict, prefix: str, ref_w: torch.Tensor):
    q = unpack_int4(sd[prefix + ".qweight"])
    zs = unpack_along_n(sd[prefix + ".qzeros"])
    s = sd[prefix + ".scales"]
    K = q.shape[0]
    g_idx = sd.get(prefix + ".g_idx")
    g_idx = g_idx.long() if g_idx is not None else torch.arange(K) // (K // s.shape[0])
    best = None
    for off in (1, 0):
        z = ((zs.int() + off) & 0xF).to(torch.uint8)
        err = (dequant(q, z, s, g_idx) - ref_w).norm().item()
        if best is None or err < best[0]:
            best = (err, off, z)
    return q, best[2], s, g_idx, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).state_dict()
    a, b = load_dir(args.ours), load_dir(args.theirs)
    b_is_quantkit = (Path(args.theirs) / "quant_config.json").exists()

    prefixes = [k[: -len(".qweight")] for k in a if k.endswith(".qweight")]
    order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def key(p):
        parts = p.split(".")
        li = int(parts[parts.index("layers") + 1])
        return li, order.index(parts[-1]) if parts[-1] in order else 99

    prefixes.sort(key=key)
    missing = [p for p in prefixes if p + ".qweight" not in b]
    if missing:
        print(f"{len(missing)} modules missing from the GPTQModel checkpoint, e.g. {missing[0]}")

    print(f"{'module':34s} {'codes equal':>11s} {'scales equal':>12s} "
          f"{'|Wa-Wb|/|W|':>12s} {'|Wa-W|/|W|':>11s} {'|Wb-W|/|W|':>11s}")
    offs = set()
    for p in prefixes:
        if p in missing:
            continue
        W = ref[p + ".weight"].float()
        qa, za, sa, ga = ours(a, p)
        if b_is_quantkit:
            qb, zb, sb, gb = ours(b, p)
        else:
            qb, zb, sb, gb, off = theirs(b, p, W)
            offs.add(off)
        Wa, Wb = dequant(qa, za, sa, ga), dequant(qb, zb, sb, gb)
        n = W.norm().item()
        codes = (qa == qb).float().mean().item()
        scal = (sa.float() == sb.float()).float().mean().item() if sa.shape == sb.shape else float("nan")
        name = p.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "")
        print(f"{name:34s} {codes:11.4f} {scal:12.4f} {(Wa - Wb).norm().item() / n:12.4g} "
              f"{(Wa - W).norm().item() / n:11.4g} {(Wb - W).norm().item() / n:11.4g}")
    if offs:
        print(f"\nGPTQModel zero-point offset detected: {sorted(offs)} (1 = stored as zero - 1)")


if __name__ == "__main__":
    main()
