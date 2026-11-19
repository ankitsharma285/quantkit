"""Record the environment with every result, so numbers stay comparable."""
from __future__ import annotations

import platform
import subprocess

import torch


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for pkg in ("transformers", "triton", "datasets"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            q = "driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,temperature.gpu"
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            info["nvidia_smi"] = dict(zip(q.split(","), [s.strip() for s in out.split(",")]))
        except Exception:
            info["nvidia_smi"] = None
    return info
