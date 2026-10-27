#!/usr/bin/env python3
"""
Per-run table for Phase 2: epochs trained, in-loop best epoch (selected by the
OLD misaligned validator), early-stop flag, fixed-geometry mAP50-95 for best.pt
and last.pt, and measured wall-clock per epoch (results.csv 'time' column,
cumulative seconds, includes the in-loop val pass). Appends to phase2_report.md
and writes phase2_runs.csv.

    .venv/bin/python -I diagnostics/presubmission/phase2_runs.py
"""
import csv
import os
import statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT = os.path.join(ROOT, "diagnostics", "presubmission")
GRID = os.path.join(ROOT, "diagnostics", "step5_retrain", "runs")
SEEDS = os.path.join(ROOT, "diagnostics", "pilot_seeds", "runs")
EPOCHS, PATIENCE = 50, 20

reval = {}
for r in csv.DictReader(open(os.path.join(OUT, "reval_fixed_geometry.csv"))):
    reval[(r["run"], int(r["seed"]), r["ckpt"])] = float(r["mAP50-95_new"])

runs = [(name, 0, os.path.join(GRID, name)) for name in sorted(os.listdir(GRID)) if name.startswith("abl_")]
for s in range(1, 8):
    for name in ("abl_cbam_rgvqf_proposed", "abl_cbam_no_vqc"):
        runs.append((name, s, os.path.join(SEEDS, f"seed{s}", name)))

rows = []
for name, seed, d in runs:
    res = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(os.path.join(d, "results.csv")))]
    m = [float(r["metrics/mAP50-95(B)"]) for r in res]
    t = [float(r["time"]) for r in res]
    bi = max(range(len(m)), key=lambda i: m[i])
    rows.append({
        "run": name, "seed": seed, "epochs_trained": len(res), "best_epoch_inloop": int(float(res[bi]["epoch"])),
        "early_stopped": len(res) < EPOCHS, "inloop_best_mAP50-95_old_geom": round(m[bi], 4),
        "best_mAP50-95_fixed": round(reval[(name, seed, "best.pt")], 4),
        "last_mAP50-95_fixed": round(reval[(name, seed, "last.pt")], 4),
        "last_minus_best": round(reval[(name, seed, "last.pt")] - reval[(name, seed, "best.pt")], 4),
        "sec_per_epoch": round(t[-1] / len(t), 1),
    })

with open(os.path.join(OUT, "phase2_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

L = ["\n## Per-run training history (in-loop selection used the OLD misaligned validator)\n\n",
     "| run | seed | epochs trained | best epoch (in-loop) | early-stopped (<50) | best.pt mAP50-95 (fixed) | last.pt mAP50-95 (fixed) | last − best | s/epoch |\n",
     "|---|---|---|---|---|---|---|---|---|\n"]
for r in rows:
    L.append(f"| {r['run']} | {r['seed']} | {r['epochs_trained']} | {r['best_epoch_inloop']} | "
             f"{'yes' if r['early_stopped'] else 'no'} | {r['best_mAP50-95_fixed']:.4f} | {r['last_mAP50-95_fixed']:.4f} | "
             f"{r['last_minus_best']:+.4f} | {r['sec_per_epoch']} |\n")
es = sum(r["early_stopped"] for r in rows)
L.append(f"\n- Early-stopped before epoch 50: {es}/{len(rows)} runs. "
         f"Median epochs trained: {st.median(r['epochs_trained'] for r in rows)}; "
         f"median in-loop best epoch: {st.median(r['best_epoch_inloop'] for r in rows)}.\n")
L.append(f"- last.pt − best.pt (fixed geometry): mean {st.mean(r['last_minus_best'] for r in rows):+.4f}, "
         f"range {min(r['last_minus_best'] for r in rows):+.4f} … {max(r['last_minus_best'] for r in rows):+.4f}.\n")
by = {}
for r in rows:
    by.setdefault(r["run"], []).append(r["sec_per_epoch"])
L.append("- Measured wall-clock per epoch (train + in-loop val, MPS), median per config: "
         + "; ".join(f"{k} {st.median(v):.0f}s" for k, v in sorted(by.items())) + "\n")
open(os.path.join(OUT, "phase2_report.md"), "a").write("".join(L))
print("".join(L))
