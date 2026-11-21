"""Practical DRAM bandwidth of this GPU, to set a realistic roofline.

    python scripts/bw_ceiling.py --out results/phase5/bw_ceiling.json

The spec-sheet number (L4: 300 GB/s) is a theoretical peak. This measures what
real kernels reach, on buffers far larger than L2 (L4: 48 MB):
  copy        dst.copy_(src): reads and writes, counted as 2x the buffer
  read        src.sum(): read-only streaming reduction
  gemv fp16   cuBLAS F.linear with M = 1 on an 8192 x 8192 fp16 weight
The best of these is the practical ceiling that "% of peak" should refer to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from quantkit.env import env_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=512, help="buffer size in MB")
    ap.add_argument("--spec-gbps", type=float, default=300.0)
    ap.add_argument("--out", default="results/phase5/bw_ceiling.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    n = args.mb * 2**20 // 2
    src = torch.randn(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    W = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    x = torch.randn(1, 8192, device="cuda", dtype=torch.float16)

    tests = {
        "copy": (lambda: dst.copy_(src), 2 * src.numel() * 2),
        "read": (lambda: src.sum(), src.numel() * 2),
        "gemv_fp16_8192x8192": (lambda: F.linear(x, W), W.numel() * 2),
    }
    results = {}
    for name, (fn, nbytes) in tests.items():
        ms = do_bench(fn, warmup=50, rep=500, return_mode="median")
        gbps = nbytes / (ms * 1e-3) / 1e9
        results[name] = {"ms": ms, "bytes": nbytes, "gbps": gbps, "pct_spec": 100 * gbps / args.spec_gbps}
        print(f"{name:22s} {gbps:7.1f} GB/s  ({100 * gbps / args.spec_gbps:5.1f}% of {args.spec_gbps:g} spec)")
    best = max(r["gbps"] for r in results.values())
    print(f"practical ceiling: {best:.1f} GB/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "practical_gbps": best, "spec_gbps": args.spec_gbps,
                               "env": env_info()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
