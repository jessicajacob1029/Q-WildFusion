#!/usr/bin/env python3
"""
Phase 2 analysis of diagnostics/presubmission/reval_fixed_geometry.csv:
  - max |change| in mAP50-95 (rows that have an old value)
  - 10-config ranking by mAP50-95, old vs new (best.pt and last.pt)
  - 8-seed paired test, abl_cbam_rgvqf_proposed vs abl_cbam_no_vqc (paired t + Wilcoxon),
    on new best.pt and new last.pt numbers (seed 0 = the grid run)
Writes phase2_report.md next to the CSV.

    .venv/bin/python -I diagnostics/presubmission/phase2_analyze.py
"""
import csv
import os
import statistics as st

from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(OUT, "reval_fixed_geometry.csv"))))
f = lambda v: float(v) if v not in ("", None) else None  # noqa: E731
for r in rows:
    r["seed"] = int(r["seed"])
    for k in ("mAP50_old", "mAP50_new", "mAP50-95_old", "mAP50-95_new", "P", "R"):
        r[k] = f(r[k])

L = ["# Phase 2 — re-validation with fixed val geometry (768x960, pad 0)\n"]
L.append(f"Rows: {len(rows)} / 48 expected.\n")

# ---- max change ----
ch = [(abs(r["mAP50-95_new"] - r["mAP50-95_old"]), r) for r in rows if r["mAP50-95_old"] is not None]
deltas = [r["mAP50-95_new"] - r["mAP50-95_old"] for _, r in ch]
mx, mr = max(ch, key=lambda t: t[0])
L.append("## Change in mAP50-95 (rows with a pre-fix value)\n")
L.append(f"- n = {len(ch)}; mean change = {st.mean(deltas):+.4f}; min = {min(deltas):+.4f}; max = {max(deltas):+.4f}\n")
L.append(f"- max |change| = {mx:.4f} ({mr['run']} seed {mr['seed']} {mr['ckpt']}: "
         f"{mr['mAP50-95_old']:.4f} -> {mr['mAP50-95_new']:.4f})\n")

# ---- ranking ----
grid = [r for r in rows if r["seed"] == 0]
for ckpt in ("best.pt", "last.pt"):
    g = [r for r in grid if r["ckpt"] == ckpt]
    if len(g) < 2:
        L.append(f"\n## 10-config ranking, {ckpt}: only {len(g)} rows so far\n")
        continue
    old_rank = sorted(g, key=lambda r: -r["mAP50-95_old"])
    new_rank = sorted(g, key=lambda r: -r["mAP50-95_new"])
    orank = {r["run"]: i + 1 for i, r in enumerate(old_rank)}
    rho = stats.spearmanr([orank[r["run"]] for r in new_rank], list(range(1, len(new_rank) + 1))).statistic
    L.append(f"\n## 10-config ranking, {ckpt} (by mAP50-95)\n\n")
    L.append(f"Ranking changed: **{[r['run'] for r in old_rank] != [r['run'] for r in new_rank]}**; "
             f"Spearman rho(old, new) = {rho:.3f}\n\n")
    L.append("| new rank | config | mAP50 new | mAP50-95 new | P@.25 | R@.25 | old rank | mAP50-95 old |\n|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(new_rank):
        L.append(f"| {i + 1} | {r['run']} | {r['mAP50_new']:.4f} | {r['mAP50-95_new']:.4f} | {r['P']:.3f} | {r['R']:.3f} | "
                 f"{orank[r['run']]} | {r['mAP50-95_old']:.4f} |\n")
    vals = [r["mAP50-95_new"] for r in g]
    L.append(f"\nSpread of new mAP50-95 across configs: {min(vals):.4f}–{max(vals):.4f} (range {max(vals) - min(vals):.4f})\n")

# ---- paired seeds ----
P_, B_ = "abl_cbam_rgvqf_proposed", "abl_cbam_no_vqc"
L.append("\n## 8-seed paired test: proposed (cbam+rgvqf) vs baseline (cbam+add)\n")
for ckpt in ("best.pt", "last.pt"):
    for metric in ("mAP50-95_new", "mAP50_new"):
        d = {(r["run"], r["seed"]): r[metric] for r in rows if r["ckpt"] == ckpt}
        seeds = [s for s in range(8) if (P_, s) in d and (B_, s) in d]
        if len(seeds) < 2:
            L.append(f"\n- {ckpt} {metric}: insufficient seeds ({seeds})\n")
            continue
        p = [d[(P_, s)] for s in seeds]
        b = [d[(B_, s)] for s in seeds]
        diff = [x - y for x, y in zip(p, b)]
        t = stats.ttest_rel(p, b)
        try:
            w = stats.wilcoxon(p, b).pvalue
        except ValueError as e:
            w = f"n/a ({e})"
        sd = st.stdev(diff)
        L.append(f"\n### {ckpt} — {metric.replace('_new', '')} (n={len(seeds)}, seeds {seeds})\n\n")
        L.append("| seed | proposed | baseline | diff |\n|---|---|---|---|\n")
        for s, x, y in zip(seeds, p, b):
            L.append(f"| {s} | {x:.4f} | {y:.4f} | {x - y:+.4f} |\n")
        L.append(f"\n- proposed {st.mean(p):.4f} ± {st.stdev(p):.4f}; baseline {st.mean(b):.4f} ± {st.stdev(b):.4f}\n")
        L.append(f"- mean diff {st.mean(diff):+.4f} (SD {sd:.4f}); paired t = {t.statistic:.3f}, p = {t.pvalue:.4f}; "
                 f"Wilcoxon p = {w if isinstance(w, str) else f'{w:.4f}'}; "
                 f"Cohen's dz = {st.mean(diff) / sd if sd else float('nan'):.3f}; "
                 f"{sum(x > 0 for x in diff)}/{len(diff)} seeds favour proposed\n")

open(os.path.join(OUT, "phase2_report.md"), "w").write("".join(L))
print("".join(L))
