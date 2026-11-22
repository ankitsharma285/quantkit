"""Zero-shot downstream accuracy with lm-evaluation-harness.

    pip install -e '.[eval]'
    python scripts/eval_tasks.py --configs fp16 w4:ckpt/rtn-g128 w4:ckpt/gptq-ao-g128 \
        w4+lmh:ckpt/gptq-ao-g128

Config strings as in quantkit/runtime.py. Quantized models run on the Triton
backend (its outputs match the reference path; see tests and Phase 4-6).
Reports acc_norm where the task defines it (ARC, HellaSwag, PIQA) and acc
otherwise (WinoGrande), with lm-eval's standard errors, plus the unweighted mean.
--limit N evaluates the first N examples per task for a quick check.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from quantkit.env import env_info
from quantkit.runtime import load_config, short_name

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def headline(task_res: dict) -> tuple[str, float, float | None]:
    for metric in ("acc_norm", "acc"):
        if f"{metric},none" in task_res:
            return metric, task_res[f"{metric},none"], task_res.get(f"{metric}_stderr,none")
    raise KeyError(f"no acc metric in {sorted(task_res)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/phase6/tasks_l4.json")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge into an existing file, so configs can be run in separate sessions.
    record = json.loads(out.read_text()) if out.exists() else {"runs": {}}

    for config in args.configs:
        name = short_name(config)
        t0 = time.perf_counter()
        model = load_config(config, args.model)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
        res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks, num_fewshot=0, limit=args.limit,
                                      log_samples=False, bootstrap_iters=1000)
        per_task = {}
        for task in args.tasks:
            metric, value, stderr = headline(res["results"][task])
            per_task[task] = {"metric": metric, "value": value, "stderr": stderr,
                              "all": {k: v for k, v in res["results"][task].items()
                                      if isinstance(v, (int, float))}}
        mean = sum(t["value"] for t in per_task.values()) / len(per_task)
        record["runs"][name] = {"config": config, "tasks": per_task, "mean": mean, "limit": args.limit,
                                "minutes": (time.perf_counter() - t0) / 60}
        cells = "  ".join(f"{t} {100 * v['value']:.1f}±{100 * (v['stderr'] or 0):.1f}"
                          for t, v in per_task.items())
        print(f"{name:28s} {cells}  mean {100 * mean:.1f}  "
              f"({record['runs'][name]['minutes']:.1f} min)", flush=True)

        record.update({"tasks": args.tasks, "num_fewshot": 0, "env": env_info(),
                       "timestamp": datetime.now(timezone.utc).isoformat()})
        out.write_text(json.dumps(record, indent=2))
        lm = model = None
        gc.collect()
        torch.cuda.empty_cache()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
