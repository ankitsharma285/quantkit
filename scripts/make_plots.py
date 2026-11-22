"""Regenerate the README figures from the saved results JSONs.

    python scripts/make_plots.py --results results --out results/plots

Reads (paths relative to --results; missing inputs skip their figure):
    phase5/bench_l4_v2.json, phase5/bw_ceiling.json   kernel speedup vs M, bandwidth at M = 1
    phase4/bench_l4.json                              Phase 4 bandwidth at M = 1
    phase6_lmh/decode_l4_lmh.json                     GPU time per decoded token
    **/*.json with a "perplexity" field               the GPTQ investigation (Phase 2)

Figures are static PNGs for the README; the README tables carry the exact numbers.
Colors: the first four slots of a categorical palette validated for color-vision
deficiency (dataviz skill: lightness band, chroma, CVD and normal-vision separation),
always paired with direct labels.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
LAYERS = ["q/o_proj", "k/v_proj", "gate/up_proj", "down_proj"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10,
})


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _median_us(rows, shape, M, impl):
    ts = [r["us"] for r in rows if r["shape"] == shape and r["M"] == M and r["impl"] == impl]
    return statistics.median(ts) if ts else None


def _gbps(rows, shape, M, impl):
    ts = [r["gbps"] for r in rows if r["shape"] == shape and r["M"] == M and r["impl"] == impl]
    return statistics.median(ts) if ts else None


def _legend_below(ax, ncols=4, handles=None, labels=None) -> None:
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=ncols, handlelength=1.6)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def speedup_vs_m(bench, out: Path) -> None:
    rows = bench["rows"]
    Ms = sorted({r["M"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(Ms[0] / 6, 0.96, "fp16 cuBLAS = 1×", color=MUTED, fontsize=8, ha="left", va="top")
    for color, shape in zip(SERIES, LAYERS):
        xs, ys = [], []
        for M in Ms:
            fp, q = _median_us(rows, shape, M, "fp16"), _median_us(rows, shape, M, "auto")
            if fp and q:
                xs.append(M)
                ys.append(fp / q)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=shape)
        ax.annotate(f"{shape}  {ys[0]:.2f}×", (xs[0], ys[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms, [str(m) for m in Ms])
    ax.set_xlim(Ms[0] / 6.5, Ms[-1] * 1.3)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("tokens per call (M)")
    ax.set_ylabel("speedup over fp16")
    ax.set_title("W4A16 Triton kernel vs fp16 cuBLAS, Llama-3.2-1B layer shapes, L4")
    _legend_below(ax)
    _save(fig, out, "kernel_speedup_vs_m.png")


def bandwidth_m1(v4, v5, ceiling, out: Path) -> None:
    impls = [("fp16 cuBLAS (fp16 weights)", v5["rows"], "fp16"),
             ("W4A16, Phase 4", v4["rows"] if v4 else None, "auto"),
             ("W4A16, Phase 5", v5["rows"], "auto")]
    impls = [i for i in impls if i[1] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    h = 0.8 / len(impls)
    for j, (color, (label, rows, impl)) in enumerate(zip(SERIES, impls)):
        for i, shape in enumerate(LAYERS):
            g = _gbps(rows, shape, 1, impl)
            y = i + (j - (len(impls) - 1) / 2) * h
            ax.barh(y, g, height=h * 0.85, color=color, label=label if i == 0 else None)
            ax.text(g + 3, y, f"{g:.0f}", va="center", fontsize=8, color=INK2)
    if ceiling:
        ax.axvline(ceiling, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(ceiling - 2, -0.62, f"measured ceiling\n{ceiling:.0f} GB/s", color=MUTED,
                fontsize=8, ha="right", va="bottom")
    ax.set_yticks(range(len(LAYERS)), LAYERS)
    ax.set_ylim(len(LAYERS) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (ceiling or 250) * 1.1)
    ax.set_xlabel("achieved DRAM bandwidth at M = 1 (GB/s)")
    ax.set_title("Bandwidth at decode (M = 1): how close each kernel gets to the hardware")
    _legend_below(ax, ncols=3)
    _save(fig, out, "kernel_bandwidth_m1.png")


CATS = [("int4_matmul", "int4 matmuls"), ("other_matmul", "fp16 matmuls (cuBLAS)"),
        ("attention", "attention"), ("other", "other kernels")]
NAMES = {"fp16": "fp16", "w4+fuse": "W4A16, fused", "w4+lmh+fuse": "W4A16, fused,\n+ int4 lm_head"}


def gpu_breakdown(decode, out: Path) -> None:
    rows = []
    for r in decode["results"]:
        key = r["config"].split(":")[0]
        if key in NAMES:
            rows.append((NAMES[key], r))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, (name, r) in enumerate(rows):
        left = 0.0
        for color, (cat, label) in zip(SERIES, CATS):
            v = r["gpu_ms_breakdown"].get(cat, 0.0)
            if v <= 0:
                continue
            ax.barh(i, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=1.5,
                    label=label if i == 0 or label not in ax.get_legend_handles_labels()[1] else None)
            if v >= 0.9:
                ax.text(left + 0.12, i, f"{v:.2f}", va="center", ha="left", fontsize=8,
                        color="#0b0b0b" if color == SERIES[3] else "#ffffff")
            left += v
        tok_s = r.get("graph_tok_s")
        ax.text(left + 0.15, i, f"{left:.2f} ms" + (f"  ·  {tok_s:.1f} tok/s" if tok_s else ""),
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(sum(r["gpu_ms_breakdown"].values()) for _, r in rows) * 1.42)
    ax.set_xlabel("GPU time per decoded token (ms)")
    ax.set_title("Where decode time goes (batch 1, CUDA graphs)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lbl) for _, lbl in CATS if lbl in labels]
    _legend_below(ax, 4, [handles[k] for k in order], [labels[k] for k in order])
    _save(fig, out, "decode_gpu_breakdown.png")


XMIN = 6.5  # left edge of the log-scale perplexity axis

GPTQ_RUNS = [
    ("gptq-w4-g128", "GPTQ, paper settings"),
    ("gptq-w4-g128-damp0.05", "GPTQ, 5% damping"),
    ("gptq-w4-g128-sg", "GPTQ, static groups"),
    ("gptq-w4-g128-skip1", "GPTQ, first token out of H"),
    ("gptq-w4-g128-sg-skip1", "GPTQ, static groups,\nfirst token out of H"),
    ("gptq-w4-g128-ao", "GPTQ, act-order (3 seeds)"),
]


def gptq_investigation(results: Path, out: Path) -> None:
    ppl = {}
    for p in sorted(results.rglob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        wt = r.get("perplexity", {}).get("wikitext2", {}) if isinstance(r, dict) else {}
        # Full evaluations only (141 windows); smoke tests run on a handful of windows.
        if wt.get("ppl") is not None and wt.get("windows", 0) >= 100:
            ppl.setdefault(r["label"], wt["ppl"])
    seeds = [ppl[k] for k in ("gptq-w4-g128-ao", "gptq-w4-g128-ao-seed1", "gptq-w4-g128-ao-seed2") if k in ppl]
    if seeds:
        ppl["gptq-w4-g128-ao"] = statistics.mean(seeds)
    runs = [(lbl, name, ppl[lbl]) for lbl, name in GPTQ_RUNS if lbl in ppl]
    if not runs or "rtn-w4-g128" not in ppl:
        print("skip gptq_investigation: missing results")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rtn, fp16 = ppl["rtn-w4-g128"], ppl.get("fp16")
    for i, (_, name, v) in enumerate(runs):
        broken = v > rtn
        ax.barh(i, v, height=0.55, color=SERIES[1] if broken else SERIES[0])
        if broken:
            ax.text(v * 1.04, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        else:  # short bars: print the value at the bar's start, clear of the reference lines
            ax.text(XMIN * 1.03, i, f"{v:.2f}", va="center", ha="left", fontsize=8.5, color="#ffffff")
    for x, label, ha in ((fp16, f"fp16 {fp16:.2f}" if fp16 else None, "right"), (rtn, f"RTN {rtn:.2f}", "left")):
        if x and label:
            ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(x * (0.985 if ha == "right" else 1.015), -0.62, label, color=MUTED, fontsize=8,
                    ha=ha, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([8, 10, 15, 20, 40, 80], ["8", "10", "15", "20", "40", "80"])
    ax.minorticks_off()
    ax.set_xlim(XMIN, max(v for *_, v in runs) * 1.5)
    ax.set_yticks(range(len(runs)), [n for _, n, _ in runs])
    ax.set_ylim(len(runs) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WikiText-2 perplexity, log scale (lower is better)")
    ax.set_title("Why vanilla GPTQ fails on Llama-3.2-1B, and what fixes it (W4 g128)")
    _save(fig, out, "gptq_investigation.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    v5 = _load(res / "phase5" / "bench_l4_v2.json")
    v4 = _load(res / "phase4" / "bench_l4.json")
    ceil = _load(res / "phase5" / "bw_ceiling.json")
    decode = _load(res / "phase6_lmh" / "decode_l4_lmh.json")
    if v5:
        speedup_vs_m(v5, out)
        bandwidth_m1(v4, v5, ceil["practical_gbps"] if ceil else None, out)
    else:
        print("skip kernel figures: phase5/bench_l4_v2.json not found")
    if decode:
        gpu_breakdown(decode, out)
    else:
        print("skip decode figure: phase6_lmh/decode_l4_lmh.json not found")
    gptq_investigation(res, out)


if __name__ == "__main__":
    main()
