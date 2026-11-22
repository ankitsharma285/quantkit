"""End-to-end batch-1 greedy decode: fp16 vs W4A16, with and without fused projections.

    python scripts/bench_decode.py --configs fp16 fp16+fuse w4:ckpt/gptq-ao w4+fuse:ckpt/gptq-ao

Each config is "fp16" or "w4:<packed checkpoint dir>", optionally with "+fuse"
(q/k/v and gate/up fused). For each config it measures:

  prefill ms      first token for a --prompt-len prompt (time to first token)
  eager tok/s     decode with normal kernel launches (Python + launch overhead included)
  graph tok/s     each decode step replayed from a CUDA graph (launch overhead removed)
  gpu ms/token    sum of GPU kernel time per decode step (torch profiler), split into
                  int4 matmuls, other matmuls (cuBLAS: fp16 linears and lm_head),
                  attention, and everything else; the floor graph mode can approach
  weights GB      parameter + buffer bytes; peak GB: max allocated during decode

At batch 1 a 1B model in eager PyTorch can be bound by the CPU launching kernels
rather than by the GPU; comparing eager, graph and GPU time shows which regime
each config is in. Greedy outputs of eager and graph decoding are compared
token by token as a correctness check.
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.autograd import DeviceType
from transformers import AutoModelForCausalLM

from quantkit.decode import Decoder
from quantkit.env import env_info
from quantkit.fuse import fuse_projections
from quantkit.model import load_quantized
from quantkit.quant.qlinear import set_backend


def load(config: str, model_id: str):
    base, _, path = config.partition(":")
    fuse = base.endswith("+fuse")
    kind = base.removesuffix("+fuse")
    if kind == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).cuda().eval()
    elif kind == "w4":
        model = load_quantized(model_id, path, "cuda")
        set_backend(model, "triton")
    else:
        raise ValueError(f"bad config {config!r}")
    if fuse:
        fuse_projections(model)
    return model


def nbytes(model) -> int:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


def kernel_category(name: str) -> str:
    n = name.lower()
    if "w4a16" in n:
        return "int4_matmul"
    if any(s in n for s in ("flash", "fmha", "attention", "sdpa", "efficient")):
        return "attention"
    if any(s in n for s in ("gemm", "gemv", "cutlass", "sm80_", "sm89_", "ampere", "cublas", "xmma")):
        return "other_matmul"
    return "other"


def timed(fn) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def run_config(config, args, prompt) -> dict:
    model = load(config, args.model)
    d = Decoder(model, args.prompt_len, args.new_tokens)
    res = {"config": config, "weights_gb": nbytes(model) / 1e9}

    # Warm-up: compiles and autotunes the Triton kernels for M = prompt_len and M = 1.
    for _ in range(2):
        d.generate(prompt)
    torch.cuda.synchronize()

    res["prefill_ms"] = sorted(timed(lambda: d.prefill(prompt)) for _ in range(5))[2]

    torch.cuda.reset_peak_memory_stats()
    eager_tokens = None
    times = []
    for _ in range(args.repeats):
        tok = d.prefill(prompt)
        torch.cuda.synchronize()
        out = []

        def loop():
            nonlocal tok
            for t in range(args.new_tokens - 1):
                tok = d.step(tok, t)
                out.append(tok)
        times.append(timed(loop))
        eager_tokens = torch.cat(out, dim=1)
    res["eager_ms_per_token"] = sorted(times)[len(times) // 2] / (args.new_tokens - 1)
    res["eager_tok_s"] = 1e3 / res["eager_ms_per_token"]
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # GPU kernel time per decode step.
    tok = d.prefill(prompt)
    torch.cuda.synchronize()
    steps = min(32, args.new_tokens - 1)
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        for t in range(steps):
            tok = d.step(tok, t)
        torch.cuda.synchronize()
    per_cat = defaultdict(float)
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            per_cat[kernel_category(e.name)] += e.time_range.elapsed_us()
    res["gpu_ms_per_token"] = sum(per_cat.values()) / steps / 1e3 if per_cat else None
    res["gpu_ms_breakdown"] = {k: v / steps / 1e3 for k, v in sorted(per_cat.items())}

    # CUDA graph replay.
    try:
        d.prefill(prompt)
        d.capture()
        times = []
        for _ in range(args.repeats):
            tok = d.prefill(prompt)
            torch.cuda.synchronize()
            out = []

            def gloop():
                nonlocal tok
                for t in range(args.new_tokens - 1):
                    tok = d.graph_step(tok, t)
                    out.append(tok)
            times.append(timed(gloop))
            graph_tokens = torch.cat(out, dim=1)
        res["graph_ms_per_token"] = sorted(times)[len(times) // 2] / (args.new_tokens - 1)
        res["graph_tok_s"] = 1e3 / res["graph_ms_per_token"]
        res["graph_matches_eager"] = float((graph_tokens == eager_tokens).float().mean())
    except Exception as ex:  # report and carry on; eager and GPU-time numbers still stand
        res["graph_error"] = f"{type(ex).__name__}: {str(ex)[:300]}"

    del d, model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/phase6/decode_l4.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    prompt = torch.randint(0, 128000, (1, args.prompt_len),
                           generator=torch.Generator().manual_seed(0)).cuda()
    results = []
    for c in args.configs:
        r = run_config(c, args, prompt)
        results.append(r)
        g = f"{r['graph_tok_s']:6.1f}" if "graph_tok_s" in r else "  fail"
        bd = "  ".join(f"{k} {v:.2f}" for k, v in r["gpu_ms_breakdown"].items())
        gpu = f"{r['gpu_ms_per_token']:.2f}" if r["gpu_ms_per_token"] is not None else "n/a"
        print(f"{c:28s} weights {r['weights_gb']:.2f} GB  peak {r['peak_gb']:.2f} GB  prefill {r['prefill_ms']:6.1f} ms"
              f"  eager {r['eager_tok_s']:6.1f} tok/s  graph {g} tok/s  gpu {gpu} ms/tok"
              f"  [{bd}]", flush=True)
        if "graph_error" in r:
            print(f"    graph capture failed: {r['graph_error']}")
        elif r["graph_matches_eager"] < 1.0:
            print(f"    note: graph tokens match eager on {100 * r['graph_matches_eager']:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "prompt_len": args.prompt_len,
                               "new_tokens": args.new_tokens, "repeats": args.repeats,
                               "env": env_info(), "timestamp": datetime.now(timezone.utc).isoformat()},
                              indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
