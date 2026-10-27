#!/usr/bin/env python3
"""
Phase 2: re-validate existing best.pt / last.pt with the fixed val geometry
(RGBTDetectionTrainer.build_dataset, val pad=0.0 -> 768x960, see Phase 1).

Calls run_step5_retrain.revalidate_checkpoint() UNMODIFIED (trainer.validate(),
corrected RGBTDetectionValidator, conf 0.001, iou 0.7, imgsz 960, rect, batch 4,
P/R read at conf 0.25). Only its scratch dir is redirected (rsr.OUT_DIR).
Checkpoints are only read. No training.

Resumable: rows already in reval_fixed_geometry.csv are skipped.

    caffeinate -i -s .venv/bin/python -u diagnostics/presubmission/phase2_revalidate.py \
        2>&1 | tee -a diagnostics/presubmission/phase2_revalidate.log
    # smoke: --limit 1
"""
import argparse
import csv
import inspect
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_step5_retrain as rsr  # noqa: E402
from phase3.code.rgbt_trainer import RGBTDetectionTrainer  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "presubmission")
CSV_PATH = os.path.join(OUT, "reval_fixed_geometry.csv")
DETAIL_PATH = os.path.join(OUT, "reval_fixed_geometry_detail.jsonl")
FIELDS = ["run", "seed", "ckpt", "mAP50_old", "mAP50_new", "mAP50-95_old", "mAP50-95_new", "P", "R"]
GRID_RUNS = os.path.join(ROOT, "diagnostics", "step5_retrain", "runs")
SEED_RUNS = os.path.join(ROOT, "diagnostics", "pilot_seeds", "runs")
SEED_CONFIGS = ["abl_cbam_rgvqf_proposed", "abl_cbam_no_vqc"]

rsr.OUT_DIR = OUT  # revalidate_checkpoint's _scratch_runs -> diagnostics/presubmission/_scratch_runs


def old_values():
    """(run, seed, ckpt) -> {mAP50, mAP50_95, P, R} from the pre-fix revalidations."""
    old = {}
    prog = json.load(open(os.path.join(ROOT, "diagnostics", "step5_retrain", "progress.json")))
    for key, cp in prog["configs"].items():
        name = ABLATIONS[key]["name"]
        for ckpt, k in (("best.pt", "best_metrics"), ("last.pt", "last_metrics")):
            if k in cp:
                old[(name, 0, ckpt)] = cp[k]
    # seeds 1-7: only best.pt was ever revalidated (pilot_results_n8.csv); last.pt has no old value
    for r in csv.DictReader(open(os.path.join(ROOT, "diagnostics", "pilot_seeds", "pilot_results_n8.csv"))):
        s = int(r["seed"])
        if s > 0:
            old[(r["config"], s, "best.pt")] = {k: float(r[k]) for k in ("mAP50", "mAP50_95", "P", "R")}
    return old


def jobs():
    out = []
    for key, cfg in ABLATIONS.items():
        for ckpt in ("best.pt", "last.pt"):
            out.append((cfg["name"], 0, ckpt, os.path.join(GRID_RUNS, cfg["name"], "weights", ckpt)))
    for seed in range(1, 8):
        for name in SEED_CONFIGS:
            for ckpt in ("best.pt", "last.pt"):
                out.append((name, seed, ckpt, os.path.join(SEED_RUNS, f"seed{seed}", name, "weights", ckpt)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="validate at most N new checkpoints (smoke test)")
    args = ap.parse_args()

    src = inspect.getsource(RGBTDetectionTrainer.build_dataset)
    assert "pad=0.0" in src, "fixed val geometry not present in RGBTDetectionTrainer.build_dataset -- STOP"
    print("[guardrail] RGBTDetectionTrainer.build_dataset has val pad=0.0")

    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.isfile(CSV_PATH):
        for r in csv.DictReader(open(CSV_PATH)):
            done.add((r["run"], int(r["seed"]), r["ckpt"]))
    else:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    old = old_values()
    device = rsr.get_device()
    all_jobs = jobs()
    todo = [j for j in all_jobs if (j[0], j[1], j[2]) not in done]
    print(f"[plan] {len(all_jobs)} checkpoints total, {len(done)} already done, {len(todo)} to go, device={device}")

    t_all = time.time()
    for i, (run, seed, ckpt, path) in enumerate(todo[: args.limit] if args.limit else todo):
        assert os.path.isfile(path), f"missing checkpoint {path}"
        t0 = time.time()
        print(f"\n[{i + 1}/{len(todo)}] {run} seed={seed} {ckpt} <- {path}")
        new = rsr.revalidate_checkpoint(path, device)
        o = old.get((run, seed, ckpt), {})
        row = {
            "run": run, "seed": seed, "ckpt": ckpt,
            "mAP50_old": o.get("mAP50", ""), "mAP50_new": new["mAP50"],
            "mAP50-95_old": o.get("mAP50_95", ""), "mAP50-95_new": new["mAP50_95"],
            "P": new["P"], "R": new["R"],
        }
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        with open(DETAIL_PATH, "a") as f:
            f.write(json.dumps({"run": run, "seed": seed, "ckpt": ckpt, "path": path, "new": new, "old": o,
                                "seconds": round(time.time() - t0, 1)}) + "\n")
        print(f"[{run} s{seed} {ckpt}] mAP50 {o.get('mAP50', float('nan')):.4f} -> {new['mAP50']:.4f} | "
              f"mAP50-95 {o.get('mAP50_95', float('nan')):.4f} -> {new['mAP50_95']:.4f} | "
              f"P {new['P']:.3f} R {new['R']:.3f} | {time.time() - t0:.0f}s")
    print(f"\n[done] {time.time() - t_all:.0f}s this session. Next: phase2_analyze.py")


if __name__ == "__main__":
    main()
