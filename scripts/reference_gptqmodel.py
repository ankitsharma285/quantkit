"""Reference run: quantize with GPTQModel, evaluate with *our* perplexity harness.

Parity rules, so a gap means a bug in quantkit and not a settings mismatch:
- the exact same calibration windows (quantkit.calib.get_calib_tokens, same seed),
  fed as token ids with no re-sorting, concatenation or filtering;
- vanilla GPTQ settings: asymmetric, damp 0.01 (GPTQModel 7.x defaults to 0.05),
  act_group_aware off and desc_act off unless --act-order (GPTQModel enables an
  activation-group-aware variant by default), true sequential on;
- the same perplexity code and windows as scripts/quantize_eval.py.

Install in a fresh Colab session, after the quantkit runs (GPTQModel may pull
its own torch/kernel builds):
    pip install -q gptqmodel
Written against GPTQModel 7.11 (GPTQConfig + CalibrationConfig API).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from quantkit.calib import get_calib_tokens
from quantkit.env import env_info
from quantkit.eval import perplexity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--static-groups", action="store_true")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", default="ckpt/ref-gptqmodel")
    ap.add_argument("--eval-dtype", choices=["native", "float16"], default="native",
                    help="GPTQModel loads Llama-3.2 in bfloat16; float16 casts the loaded model "
                         "so evaluation matches quantkit")
    ap.add_argument("--reuse", action="store_true",
                    help="skip quantization if --out already holds a checkpoint")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    import gptqmodel
    from gptqmodel import CalibrationConfig, GPTQConfig, GPTQModel

    tok = AutoTokenizer.from_pretrained(args.model)
    calib = get_calib_tokens(tok, "c4", args.nsamples, args.calib_seqlen, args.calib_seed)
    rows = [{"input_ids": r.tolist(), "attention_mask": [1] * r.shape[0]} for r in calib]

    cfg = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=False,
        desc_act=args.act_order,
        # Our act-order fixes group scales up front (contiguous groups, no g_idx);
        # GPTQModel only does the same with static_groups=True.
        static_groups=args.act_order or args.static_groups,
        act_group_aware=False,
        damp_percent=args.damp,
        true_sequential=True,
    )
    quant_time = 0.0
    if args.reuse and any(Path(args.out).glob("*.safetensors")):
        print(f"reusing checkpoint in {args.out}")
    else:
        t0 = time.perf_counter()
        model = GPTQModel.load(args.model, cfg)
        model.quantize(CalibrationConfig(dataset=rows, batch_size=1, sort=None))
        quant_time = time.perf_counter() - t0
        model.save(args.out)
        del model
        torch.cuda.empty_cache()

    qmodel = GPTQModel.load(args.out, device="cuda")
    hf_model = qmodel.model  # the underlying transformers module
    if args.eval_dtype == "float16":
        hf_model = hf_model.to(torch.float16)
    eval_dtype = str(next(p for p in hf_model.parameters() if p.is_floating_point()).dtype)
    print(f"evaluating in {eval_dtype}")
    label = f"ref-gptqmodel-w{args.bits}-g{args.group_size}"
    label += "-ao" if args.act_order else ("-sg" if args.static_groups else "")
    if args.damp != 0.01:
        label += f"-damp{args.damp}"
    if args.calib_seed != 0:
        label += f"-seed{args.calib_seed}"
    if args.nsamples != 128:
        label += f"-n{args.nsamples}"
    if args.eval_dtype == "float16":
        label += "-evalfp16"

    ppl = {}
    for ds in args.datasets:
        r = perplexity(hf_model, tok, ds, args.seqlen, args.max_windows, "cuda")
        ppl[ds] = r
        print(f"[{label}] {ds}: ppl={r['ppl']:.3f}")

    record = {
        "label": label,
        "model": args.model,
        "quant": {"method": "gptqmodel", "version": gptqmodel.__version__, "bits": args.bits,
                  "group_size": args.group_size, "sym": False, "damp": args.damp,
                  "act_order": args.act_order, "static_groups": args.act_order or args.static_groups,
                  "act_group_aware": False,
                  "calib_dataset": "c4", "nsamples": args.nsamples,
                  "calib_seqlen": args.calib_seqlen, "calib_seed": args.calib_seed,
                  "eval_dtype": eval_dtype},
        "perplexity": ppl,
        "quant_seconds": round(quant_time, 2),
        "env": env_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model.split('/')[-1]}_{label}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
