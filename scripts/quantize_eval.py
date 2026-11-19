"""Quantize a model and evaluate perplexity; every run writes a results JSON.

Examples:
    python scripts/quantize_eval.py --method fp16
    python scripts/quantize_eval.py --method rtn --group-size 128 --save ckpt/rtn-g128
    python scripts/quantize_eval.py --method rtn --group-size -1           # per-channel
    python scripts/quantize_eval.py --method rtn --bits 3 --fake           # 3-bit ablation
    python scripts/quantize_eval.py --load ckpt/rtn-g128                   # reload + eval
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
from quantkit.model import load_quantized, quantize_model_rtn, save_quantized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--method", choices=["fp16", "rtn"], default="fp16")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128, help="-1 = per-channel")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--fake", action="store_true", help="fake-quant fp16 weights instead of packing")
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None, help="cap windows for quick checks")
    ap.add_argument("--save", type=str, default=None, help="dir to save packed checkpoint")
    ap.add_argument("--load", type=str, default=None, help="dir of a saved packed checkpoint")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()

    if args.load:
        model = load_quantized(args.model, args.load, device)
        meta = json.loads((Path(args.load) / "quant_config.json").read_text())
        label = f"loaded:{Path(args.load).name}"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
        meta = {"method": args.method, "bits": args.bits, "group_size": args.group_size,
                "sym": args.sym, "packed": not args.fake}
        if args.method == "rtn":
            quantize_model_rtn(model, args.bits, args.group_size, args.sym, pack=not args.fake)
        label = args.method if args.method == "fp16" else (
            f"rtn-w{args.bits}-g{args.group_size}{'-sym' if args.sym else ''}{'-fake' if args.fake else ''}"
        )
    quant_time = time.perf_counter() - t0

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
        "setup_seconds": round(quant_time, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None,
        "env": env_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model.split('/')[-1]}_{label.replace(':', '_')}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
