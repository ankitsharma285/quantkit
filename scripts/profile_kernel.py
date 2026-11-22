"""Run one layer shape through one implementation, for Nsight Compute.

Autotuning and dispatch happen before the profiled region, so ncu sees only the
steady-state kernels (for the atomic paths: zero-fill, matmul and fp16 cast).

    ncu --profile-from-start off --clock-control none \\
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \\
        --section LaunchStats --section WarpStateStats \\
        python scripts/profile_kernel.py --shape gate/up_proj --M 1 --impl auto

--clock-control none profiles at the clocks do_bench sees (ncu otherwise locks
the L4 to its ~795 MHz base clock); percent-of-peak metrics are valid either way.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from quantkit.kernels import w4a16 as K4
from quantkit.quant.qlinear import QuantLinear
from quantkit.quant.rtn import quantize_rtn

SHAPES = {"q/o_proj": (2048, 2048), "k/v_proj": (512, 2048),
          "gate/up_proj": (8192, 2048), "down_proj": (2048, 8192)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=list(SHAPES), default="gate/up_proj")
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--impl", choices=["fp16", "auto", *K4.PATHS], default="auto")
    ap.add_argument("--unpack", choices=["auto", "once", "per_nibble"], default="auto",
                    help="auto: let path=auto measure it (explicit paths default to once)")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    N, K = SHAPES[args.shape]
    W = torch.randn(N, K, generator=torch.Generator().manual_seed(0)) * 0.02
    ql = QuantLinear.from_params(quantize_rtn(W, 4, 128), dtype=torch.float16).cuda()
    W16 = ql.dequantize_weight(torch.float16)
    x = torch.randn(args.M, K, device="cuda", dtype=torch.float16)
    once = {"auto": None, "once": True, "per_nibble": False}[args.unpack]

    if args.impl == "fp16":
        fn = lambda: F.linear(x, W16)  # noqa: E731
    else:
        fn = lambda: K4.w4a16_matmul(x, ql.qweight, ql.scales, ql.qzeros, 128,  # noqa: E731
                                      path=args.impl, unpack_once=once)
    for _ in range(3):  # autotune, dispatch, warm caches
        fn()
    torch.cuda.synchronize()
    if args.impl == "auto":
        print(f"auto dispatch chose: {K4.dispatch_choice(args.M, N, K, 128, once)}")

    torch.cuda.profiler.start()
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
