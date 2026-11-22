"""Quantize a model and evaluate perplexity; every run writes a results JSON.

Examples:
    python scripts/quantize_eval.py --method fp16
    python scripts/quantize_eval.py --method rtn --group-size 128 --save ckpt/rtn-g128
    python scripts/quantize_eval.py --method rtn --group-size -1           # per-channel
    python scripts/quantize_eval.py --method rtn --bits 3 --fake           # 3-bit ablation
    python scripts/quantize_eval.py --load ckpt/rtn-g128                   # reload + eval
    python scripts/quantize_eval.py --method gptq --save ckpt/gptq-g128
    python scripts/quantize_eval.py --method gptq --act-order
    python scripts/quantize_eval.py --method gptq --bits 3 --fake
    python scripts/quantize_eval.py --method gptq --nsamples 32 --calib-dataset wikitext2
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantkit.env import env_info
from quantkit.eval import perplexity
from quantkit.model import load_quantized, quantize_lm_head, quantize_model_rtn, save_quantized


def make_label(args) -> str:
    bf16 = "-bf16" if args.dtype == "bfloat16" else ""
    if args.method == "fp16":
        return "fp16" + bf16
    label = f"{args.method}-w{args.bits}-g{args.group_size}"
    if args.method == "gptq":
        if args.act_order:
            label += "-ao"
        elif args.static_groups:
            label += "-sg"
        if args.nsamples != 128:
            label += f"-n{args.nsamples}"
        if args.calib_dataset != "c4":
            label += f"-cal{args.calib_dataset}"
        if args.calib_seed != 0:
            label += f"-seed{args.calib_seed}"
        if args.no_true_sequential:
            label += "-notseq"
        if args.damp != 0.01:
            label += f"-damp{args.damp:g}"
        if args.blocksize != 128:
            label += f"-bs{args.blocksize}"
        if args.skip_first_tokens:
            label += f"-skip{args.skip_first_tokens}"
        if args.tf32:
            label += "-tf32"
    if args.sym:
        label += "-sym"
    if args.fake:
        label += "-fake"
    return label + bf16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--method", choices=["fp16", "rtn", "gptq"], default="fp16")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128, help="-1 = per-channel")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--fake", action="store_true", help="fake-quant fp16 weights instead of packing")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                    help="model dtype for calibration and evaluation (GPTQModel uses bfloat16)")
    # GPTQ
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    ap.add_argument("--calib-seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--static-groups", action="store_true",
                    help="fix group scales from the original weights (implied by --act-order)")
    ap.add_argument("--no-true-sequential", action="store_true")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmuls during quantization (numerical-sensitivity test)")
    ap.add_argument("--skip-first-tokens", type=int, default=0,
                    help="leave the first N positions of each calibration window out of the Hessians")
    # Eval / IO
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None, help="cap windows for quick checks")
    ap.add_argument("--save", type=str, default=None, help="dir to save packed checkpoint")
    ap.add_argument("--load", type=str, default=None, help="dir of a saved packed checkpoint")
    ap.add_argument("--backend", choices=["reference", "triton"], default="reference",
                    help="QuantLinear forward path used for evaluation")
    ap.add_argument("--fuse", action="store_true", help="fuse q/k/v and gate/up for evaluation")
    ap.add_argument("--quantize-lm-head", action="store_true",
                    help="also RTN-quantize lm_head (untied from the fp16 embedding); "
                         "works with --method fp16 and with --load")
    ap.add_argument("--lm-head-bits", type=int, default=4)
    ap.add_argument("--lm-head-group-size", type=int, default=128)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    extra: dict = {}

    if args.load:
        model = load_quantized(args.model, args.load, device)
        meta = json.loads((Path(args.load) / "quant_config.json").read_text())
        label = f"loaded:{Path(args.load).name}"
        quant_time = 0.0
    else:
        if args.dtype == "bfloat16" and args.method != "fp16" and not args.fake:
            raise SystemExit("--dtype bfloat16 needs --fake (packed QuantLinear stores fp16 scales)")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
        meta = {"method": args.method, "bits": args.bits, "group_size": args.group_size,
                "sym": args.sym, "packed": not args.fake, "dtype": args.dtype}
        label = make_label(args)
        t0 = time.perf_counter()
        if args.method == "rtn":
            quantize_model_rtn(model, args.bits, args.group_size, args.sym, pack=not args.fake)
        elif args.method == "gptq":
            from quantkit.calib import get_calib_tokens
            from quantkit.gptq_model import quantize_model_gptq

            calib = get_calib_tokens(tok, args.calib_dataset, args.nsamples, args.calib_seqlen,
                                     args.calib_seed)
            meta.update({
                "calib_dataset": args.calib_dataset, "nsamples": args.nsamples,
                "calib_seqlen": args.calib_seqlen, "calib_seed": args.calib_seed,
                "blocksize": args.blocksize, "damp": args.damp, "act_order": args.act_order,
                "static_groups": args.static_groups or args.act_order,
                "true_sequential": not args.no_true_sequential,
                "skip_first_tokens": args.skip_first_tokens, "tf32": args.tf32,
            })
            t0 = time.perf_counter()  # time the algorithm, not the dataset download
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            extra["layer_stats"] = quantize_model_gptq(
                model, calib, bits=args.bits, group_size=args.group_size, sym=args.sym,
                blocksize=args.blocksize, percdamp=args.damp, act_order=args.act_order,
                static_groups=args.static_groups,
                true_sequential=not args.no_true_sequential, pack=not args.fake, device=device,
                skip_first_tokens=args.skip_first_tokens,
            )
            torch.backends.cuda.matmul.allow_tf32 = False  # evaluate at full precision
        quant_time = time.perf_counter() - t0

    if args.quantize_lm_head:
        quantize_lm_head(model, args.lm_head_bits, args.lm_head_group_size, args.sym)
        label += f"-lmh{args.lm_head_bits}g{args.lm_head_group_size}"
        meta["lm_head"] = {"method": "rtn", "bits": args.lm_head_bits,
                           "group_size": args.lm_head_group_size}

    if args.backend != "reference":
        from quantkit.quant.qlinear import set_backend

        n = set_backend(model, args.backend)
        if n == 0:
            raise SystemExit("--backend triton needs a packed model (not --fake or fp16)")
        label += f"+{args.backend}"
        meta["backend"] = args.backend
        print(f"switched {n} QuantLinear modules to the {args.backend} backend")

    if args.fuse:
        if args.save:
            raise SystemExit("--fuse is an evaluation-time transform; don't combine with --save")
        from quantkit.fuse import fuse_projections

        fuse_projections(model)
        label += "+fuse"
        meta["fused"] = True

    if args.save:
        if args.fake or args.method == "fp16":
            raise SystemExit("--save only applies to packed quantized models")
        save_quantized(model, args.save, meta)

    ppl = {}
    for ds in args.datasets:
        r = perplexity(model, tok, ds, args.seqlen, args.max_windows, device)
        ppl[ds] = r
        print(f"[{label}] {ds}: ppl={r['ppl']:.3f}  ({r['windows']} x {r['seqlen']} tokens)")

    record = {
        "label": label,
        "model": args.model,
        "quant": meta,
        "perplexity": ppl,
        "quant_seconds": round(quant_time, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None,
        "env": env_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model.split('/')[-1]}_{label.replace(':', '_')}.json"
    if path.exists() and not args.load:
        print(f"note: overwriting {path}")
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
