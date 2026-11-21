"""Micro-benchmark: W4A16 Triton kernel vs fp16 cuBLAS on Llama-3.2-1B layer shapes.

    python scripts/bench_kernel.py                       # all shapes, M = 1..256
    python scripts/bench_kernel.py --ms 1 16 --rounds 5  # quick check

For each (layer shape, M) it times, on identical weights:
  fp16       F.linear with the dequantized fp16 weight (cuBLAS): the baseline
  reference  QuantLinear "reference" path: dequantize all of W, then F.linear
  splitk     Triton split-K kernel (M <= 16 only)
  gemm       Triton tiled kernel
  auto       what QuantLinear(backend="triton") dispatches to

Method: triton.testing.do_bench (flushes L2 between runs, so weights come from
DRAM as in real decode), median over `rep` ms; implementations are interleaved
across `--rounds` and the median round is reported, because the L4 throttles
under sustained load. Achieved bandwidth counts the bytes each implementation
must move (weights + activations + output), so "% of peak" is comparable across
implementations: memory-bound decode should approach the L4's ~300 GB/s.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from quantkit.env import env_info
from quantkit.kernels import w4a16 as K4
from quantkit.quant.qlinear import QuantLinear
from quantkit.quant.rtn import quantize_rtn

SHAPES = {  # name: (N, K) = (out_features, in_features)
    "q/o_proj": (2048, 2048),
    "k/v_proj": (512, 2048),
    "gate/up_proj": (8192, 2048),
    "down_proj": (2048, 8192),
}


def bytes_moved(impl: str, M: int, N: int, K: int, G: int) -> int:
    act = M * K * 2 + M * N * 2
    if impl == "fp16":
        return N * K * 2 + act
    if impl == "reference":  # read int4 + scales, write fp16 W, read it back for the GEMM
        return N * K // 2 + (K // G) * N * 3 + 2 * N * K * 2 + act
    return N * K // 2 + (K // G) * N * 3 + act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rep", type=int, default=100, help="ms of timing per do_bench call")
    ap.add_argument("--cooldown", type=float, default=1.0, help="seconds between shapes")
    ap.add_argument("--peak-gbps", type=float, default=300.0, help="L4 DRAM peak")
    ap.add_argument("--out", default="results/kernel/bench_l4.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    torch.backends.cuda.matmul.allow_tf32 = False
    G = args.group_size
    rows = []

    for shape in args.shapes:
        N, K = SHAPES[shape]
        g = torch.Generator().manual_seed(0)
        W = torch.randn(N, K, generator=g) * 0.02
        ql = QuantLinear.from_params(quantize_rtn(W, 4, G), dtype=torch.float16).cuda()
        W16 = ql.dequantize_weight(torch.float16)
        qargs = (ql.qweight, ql.scales, ql.qzeros, G)

        for M in args.ms:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            impls = {
                "fp16": lambda: F.linear(x, W16),
                "reference": lambda: F.linear(x, ql.dequantize_weight(torch.float16)),
                "gemm": lambda: K4.w4a16_matmul(x, *qargs, path="gemm"),
                "auto": lambda: K4.w4a16_matmul(x, *qargs),
            }
            if M <= K4.SMALL_M:
                impls["splitk"] = lambda: K4.w4a16_matmul(x, *qargs, path="splitk")

            # Correctness first (this call also runs the autotuner for this M, N, K).
            ref = F.linear(x.float(), W16.float())
            for name in ("gemm", "auto", "splitk"):
                if name in impls:
                    err = (impls[name]().float() - ref).abs().max().item()
                    tol = 2e-3 * ref.abs().max().item() + 1e-3
                    if err > tol:
                        raise AssertionError(f"{shape} M={M} {name}: max err {err:.3g} > {tol:.3g}")

            times = {k: [] for k in impls}
            for _ in range(args.rounds):
                for name, fn in impls.items():
                    times[name].append(do_bench(fn, warmup=25, rep=args.rep, return_mode="median"))
            for name, ts in times.items():
                ms = statistics.median(ts)
                b = bytes_moved(name, M, N, K, G)
                rows.append({
                    "shape": shape, "N": N, "K": K, "M": M, "impl": name,
                    "us": ms * 1e3, "us_rounds": [t * 1e3 for t in ts],
                    "gbps": b / (ms * 1e-3) / 1e9,
                    "pct_peak_bw": 100 * b / (ms * 1e-3) / 1e9 / args.peak_gbps,
                    "tflops": 2 * M * N * K / (ms * 1e-3) / 1e12,
                })
            fp = statistics.median(times["fp16"])
            line = "  ".join(
                f"{n} {statistics.median(t) * 1e3:7.1f}us ({fp / statistics.median(t):4.2f}x)"
                for n, t in times.items() if n != "fp16"
            )
            print(f"{shape:13s} M={M:4d}  fp16 {fp * 1e3:7.1f}us  {line}", flush=True)
        time.sleep(args.cooldown)

    tuned = {
        "splitk": {str(k): str(v) for k, v in K4._splitk_tuned.cache.items()},
        "gemm": {str(k): str(v) for k, v in K4._gemm_tuned.cache.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows": rows, "autotuned_configs": tuned, "group_size": G, "peak_gbps": args.peak_gbps,
        "env": env_info(), "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
