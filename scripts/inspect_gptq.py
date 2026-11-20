"""Per-layer diagnostics from GPTQ results JSONs.

    python scripts/inspect_gptq.py results/phase2/*gptq*.json

For each run: the modules with the largest GPTQ loss, and how lopsided each
Hessian diagonal is (max / median). A ratio in the thousands means a few input
channels carry almost all of the activation energy, which is the signature of
massive activations on attention-sink tokens.
"""
from __future__ import annotations

import json
import sys


def main(paths: list[str]) -> None:
    for path in paths:
        r = json.load(open(path))
        stats = r.get("layer_stats")
        if not stats:
            continue
        q = r["quant"]
        ppl = r["perplexity"].get("wikitext2", {}).get("ppl", float("nan"))
        print(f"\n== {r['label']}  (damp={q.get('damp')}, act_order={q.get('act_order')}, "
              f"skip_first_tokens={q.get('skip_first_tokens', 0)})  wikitext2 ppl={ppl:.3f}")
        rows = sorted(stats.items(), key=lambda kv: kv[1]["loss"], reverse=True)
        print(f"{'module':40s} {'loss':>11s} {'Hdiag max/median':>17s} {'Hdiag max':>11s}")
        for name, st in rows[:12]:
            ratio = st.get("hdiag_max", float("nan")) / max(st.get("hdiag_median", float("nan")), 1e-30)
            print(f"{name:40s} {st['loss']:11.4g} {ratio:17.4g} {st.get('hdiag_max', float('nan')):11.4g}")
        total = sum(st["loss"] for st in stats.values())
        print(f"{'total loss':40s} {total:11.4g}")


if __name__ == "__main__":
    main(sys.argv[1:])
