#!/usr/bin/env python3
"""
Q-WildFusion variance pilot: paired multi-seed, proposed vs baseline.

Context: the 10-config grid (diagnostics/step5_retrain, SEED=0) found every
decision-critical gap smaller than its own best<->last checkpoint spread --
see diagnostics/step5_retrain/ranked_grid.md's flag (b). A power analysis on
that single-seed data says the only reachable claim is proposed vs baseline,
and only with a paired multi-seed design. This script MEASURES the real
seed-to-seed variance directly (5 seeds, paired) instead of assuming a sigma.

Configs:
  abl_cbam_rgvqf_proposed  (proposed: cbam + rgvqf, entangled)
  abl_cbam_no_vqc          (baseline: cbam + add fusion)
Seeds: 0, 1, 2, 3, 4 (paired -- same seed set for both configs).

Seed 0 is REUSED from diagnostics/step5_retrain/ranked_grid.csv (not
retrained) -- only seeds 1-4 are trained here (8 new runs total).

CRITICAL per the task: seeds 1-4 are trained through the EXACT SAME entry
point as the grid -- this script imports run_step5_retrain as a module and
calls its own train_config()/revalidate_checkpoint()/get_device() functions
UNMODIFIED, only overriding the module-level SEED/RUNS_DIR/CURVES_DIR/OUT_DIR
globals those functions read at call time (Python resolves module globals
dynamically at call time, so this is a legitimate parameterization of the
existing entry point, not a reimplementation of it). This keeps every
training hyperparameter (imgsz, batch, box/cls/dfl, patience=20, epochs=50,
fraction=1.0, rect, augmentation-off flags, deterministic=True -- Ultralytics'
own default, unchanged from the grid) byte-identical to the grid, so seeds
1-4 are numerically comparable to the reused seed-0. The RUNS_DIR/CURVES_DIR/
OUT_DIR overrides exist ONLY to redirect where each seed's run and re-
validation scratch files land, so nothing under diagnostics/step5_retrain is
ever touched -- see the guardrail below.

Guardrails:
  - No model/fusion/trainer science source is edited or reimplemented here.
  - All outputs under diagnostics/pilot_seeds/. diagnostics/step5_retrain is
    read-only (only ranked_grid.csv is read, for the seed-0 rows).

Usage:
    caffeinate -i -s python -u run_pilot_seeds.py 2>&1 | tee diagnostics/pilot_seeds/run.log

    python run_pilot_seeds.py --check-only
        Guardrail checks + resolved live data config only. No training.

Resumable: diagnostics/pilot_seeds/progress.json is keyed by (config, seed)
-> pending/training/done. Re-running after a crash/reboot skips anything
already 'done'; a config/seed whose training was left mid-run resumes via
the same Ultralytics resume=True mechanism run_step5_retrain.train_config()
already uses (it detects an unfinalized last.pt at the seed's redirected
run directory and resumes it directly). Each result row is flushed to
diagnostics/pilot_seeds/pilot_results.csv immediately after that run's
best.pt is re-validated -- never held to the end.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import run_step5_retrain as rsr  # noqa: E402  (reused entry point -- see module docstring)

# ---------------------------------------------------------------------------
# Settings (per task spec -- do not shorten epochs/patience, must match the grid)
# ---------------------------------------------------------------------------
EPOCHS = 50
PATIENCE = 20
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ["abl_cbam_rgvqf_proposed", "abl_cbam_no_vqc"]
PROPOSED_NAME = "abl_cbam_rgvqf_proposed"
BASELINE_NAME = "abl_cbam_no_vqc"

assert EPOCHS == rsr.MODERATE_EPOCHS, (
    f"EPOCHS ({EPOCHS}) must match run_step5_retrain.MODERATE_EPOCHS ({rsr.MODERATE_EPOCHS}) "
    "for seed-0 comparability -- do not shorten."
)
assert PATIENCE == rsr.PATIENCE, (
    f"PATIENCE ({PATIENCE}) must match run_step5_retrain.PATIENCE ({rsr.PATIENCE}) "
    "for seed-0 comparability -- do not shorten."
)

# ABLATIONS key -> config name, reused from train_ablation_study (same source
# run_step5_retrain itself uses)
from phase3.code.train_ablation_study import ABLATIONS  # noqa: E402

NAME_TO_KEY = {cfg["name"]: key for key, cfg in ABLATIONS.items()}
for name in CONFIGS:
    assert name in NAME_TO_KEY, f"{name!r} not found in train_ablation_study.ABLATIONS"

PILOT_OUT_DIR = os.path.join(ROOT, "diagnostics", "pilot_seeds")
PILOT_RUNS_DIR = os.path.join(PILOT_OUT_DIR, "runs")
PILOT_CURVES_DIR = os.path.join(PILOT_OUT_DIR, "curves")
PROGRESS_PATH = os.path.join(PILOT_OUT_DIR, "progress.json")
RESULTS_CSV = os.path.join(PILOT_OUT_DIR, "pilot_results.csv")
ANALYSIS_MD = os.path.join(PILOT_OUT_DIR, "pilot_analysis.md")
GRID_RANKED_CSV = os.path.join(ROOT, "diagnostics", "step5_retrain", "ranked_grid.csv")

# Redirect run_step5_retrain's OUT_DIR so revalidate_checkpoint()'s internal
# scratch project (os.path.join(OUT_DIR, "_scratch_runs")) never lands under
# diagnostics/step5_retrain. RUNS_DIR/CURVES_DIR are overridden per-run below
# (they're read at call time inside train_config(), so this is safe).
rsr.OUT_DIR = PILOT_OUT_DIR


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------
def load_progress():
    if os.path.isfile(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"runs": {name: {} for name in CONFIGS}}


def save_progress(progress):
    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    os.replace(tmp, PROGRESS_PATH)


def append_result_row(row):
    write_header = not os.path.isfile(RESULTS_CSV)
    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "P", "R", "mAP50", "mAP50_95", "source"])
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Seed 0: reused from the grid, never retrained
# ---------------------------------------------------------------------------
def load_seed0_rows():
    if not os.path.isfile(GRID_RANKED_CSV):
        raise FileNotFoundError(
            f"{GRID_RANKED_CSV} not found -- cannot reuse seed 0. Run the grid first."
        )
    rows = {}
    with open(GRID_RANKED_CSV) as f:
        for r in csv.DictReader(f):
            if r["config"] in CONFIGS and r["checkpoint"] == "best.pt":
                rows[r["config"]] = {
                    "config": r["config"],
                    "seed": 0,
                    "P": float(r["P"]),
                    "R": float(r["R"]),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                    "source": "reused_from_grid",
                }
    missing = set(CONFIGS) - set(rows)
    if missing:
        raise ValueError(f"{GRID_RANKED_CSV} is missing best.pt rows for: {missing}")
    return rows


# ---------------------------------------------------------------------------
# Train + validate one (config, seed) through run_step5_retrain's own
# entry point, redirected to diagnostics/pilot_seeds/.
# ---------------------------------------------------------------------------
def train_and_validate(config_name, seed, device):
    key = NAME_TO_KEY[config_name]
    config = ABLATIONS[key]

    seed_runs_dir = os.path.join(PILOT_RUNS_DIR, f"seed{seed}")
    seed_curves_dir = os.path.join(PILOT_CURVES_DIR, f"seed{seed}")
    os.makedirs(seed_runs_dir, exist_ok=True)
    os.makedirs(seed_curves_dir, exist_ok=True)

    # Redirect the exact-entry-point function's module globals for this call only.
    rsr.SEED = seed
    rsr.RUNS_DIR = seed_runs_dir
    rsr.CURVES_DIR = seed_curves_dir

    run_dir, best_pt, last_pt = rsr.train_config(key, config, device)
    best_metrics = rsr.revalidate_checkpoint(best_pt, device)

    return {
        "config": config_name,
        "seed": seed,
        "P": best_metrics["P"],
        "R": best_metrics["R"],
        "mAP50": best_metrics["mAP50"],
        "mAP50_95": best_metrics["mAP50_95"],
        "source": "trained_by_pilot",
    }


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------
def paired_ttest(diffs):
    n = len(diffs)
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    sd_diff = math.sqrt(var)
    se = sd_diff / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else float("inf") if mean_diff != 0 else 0.0
    df = n - 1

    try:
        from scipy import stats

        p_value = float(2 * stats.t.sf(abs(t_stat), df))
        scipy_used = True
    except ImportError:
        # Hand-rolled two-sided p-value from the t distribution's survival
        # function via a series approximation is overkill for n=5; fall back
        # to a normal approximation and flag it clearly.
        p_value = float(2 * (1 - _norm_cdf(abs(t_stat))))
        scipy_used = False

    dz = mean_diff / sd_diff if sd_diff > 0 else float("inf") if mean_diff != 0 else 0.0
    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "dz": dz,
        "scipy_used": scipy_used,
    }


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def required_n_for_power(dz, alpha=0.05, power=0.80, n_max=1000):
    """Smallest n (paired t-test, two-sided) such that power >= target, at
    the MEASURED effect size dz. Uses the noncentral t distribution via
    scipy if available (exact); falls back to the standard normal
    approximation n ~= ((z_alpha/2 + z_power) / dz)^2 otherwise."""
    if dz == 0:
        return None  # no effect measured -- no finite n reaches 80% power
    try:
        from scipy import stats

        for n in range(2, n_max + 1):
            df = n - 1
            ncp = dz * math.sqrt(n)
            crit = stats.t.ppf(1 - alpha / 2, df)
            pw = 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
            if pw >= power:
                return n
        return None  # exceeds n_max
    except ImportError:
        z_alpha = 1.959964
        z_power = 0.841621
        n = ((z_alpha + z_power) / dz) ** 2
        return max(2, math.ceil(n))


def build_analysis(rows):
    by_config = {name: {} for name in CONFIGS}
    for r in rows:
        by_config[r["config"]][r["seed"]] = r

    per_seed = []
    diffs_map50 = []
    diffs_map5095 = []
    for seed in SEEDS:
        p = by_config[PROPOSED_NAME].get(seed)
        b = by_config[BASELINE_NAME].get(seed)
        if p is None or b is None:
            continue
        d50 = p["mAP50"] - b["mAP50"]
        d5095 = p["mAP50_95"] - b["mAP50_95"]
        per_seed.append(
            {
                "seed": seed,
                "proposed_mAP50": p["mAP50"],
                "baseline_mAP50": b["mAP50"],
                "diff_mAP50": d50,
                "proposed_mAP50_95": p["mAP50_95"],
                "baseline_mAP50_95": b["mAP50_95"],
                "diff_mAP50_95": d5095,
            }
        )
        diffs_map50.append(d50)
        diffs_map5095.append(d5095)

    stats_map50 = paired_ttest(diffs_map50) if len(diffs_map50) >= 2 else None
    stats_map5095 = paired_ttest(diffs_map5095) if len(diffs_map5095) >= 2 else None

    # Primary decision metric: mAP50-95 (threshold-integrated, matches the
    # fitness weighting used throughout this project: 0.1*mAP50 + 0.9*mAP50-95).
    primary = stats_map5095
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs_map5095]
    sign_consistent = len(set(signs)) == 1 and signs[0] != 0

    verdict = "PARITY"
    n_required = None
    if primary and primary["p_value"] < 0.05 and primary["mean_diff"] > 0:
        verdict = "SIGNIFICANT"
    elif sign_consistent and primary and abs(primary["dz"]) >= 0.5:
        verdict = "UNDERPOWERED-BUT-PROMISING"
        n_required = required_n_for_power(primary["dz"])
    else:
        verdict = "PARITY"

    return {
        "per_seed": per_seed,
        "stats_map50": stats_map50,
        "stats_map50_95": stats_map5095,
        "sign_consistent": sign_consistent,
        "verdict": verdict,
        "n_required": n_required,
    }


def write_analysis_md(analysis):
    lines = []
    lines.append("# Variance pilot -- paired multi-seed, proposed vs baseline\n\n")
    lines.append(f"Proposed: `{PROPOSED_NAME}`  Baseline: `{BASELINE_NAME}`  Seeds: {SEEDS}\n\n")
    lines.append(
        "Seed 0 reused from `diagnostics/step5_retrain/ranked_grid.csv` (not retrained); "
        "seeds 1-4 trained here through `run_step5_retrain.train_config()`/`revalidate_checkpoint()` "
        "unmodified, EPOCHS=50 PATIENCE=20 (byte-identical to the grid's hyperparameters).\n\n"
    )
    lines.append("## Per-seed values and paired diffs\n\n")
    lines.append(
        "| seed | proposed mAP50 | baseline mAP50 | diff mAP50 | proposed mAP50-95 | "
        "baseline mAP50-95 | diff mAP50-95 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in analysis["per_seed"]:
        lines.append(
            f"| {r['seed']} | {r['proposed_mAP50']:.4f} | {r['baseline_mAP50']:.4f} | "
            f"{r['diff_mAP50']:+.4f} | {r['proposed_mAP50_95']:.4f} | {r['baseline_mAP50_95']:.4f} | "
            f"{r['diff_mAP50_95']:+.4f} |\n"
        )

    def stats_block(title, s):
        b = [f"\n## {title}\n\n"]
        if s is None:
            b.append("- insufficient paired seeds to compute (need >= 2)\n")
            return "".join(b)
        b.append(f"- n = {s['n']}\n")
        b.append(f"- mean diff = {s['mean_diff']:+.4f}\n")
        b.append(f"- SD of diffs (measured seed-to-seed variance) = {s['sd_diff']:.4f}\n")
        b.append(f"- paired t = {s['t_stat']:.3f}, df = {s['df']}\n")
        b.append(f"- two-sided p-value = {s['p_value']:.4f}" + ("" if s["scipy_used"] else " (normal approx, scipy unavailable)") + "\n")
        b.append(f"- Cohen's dz = {s['dz']:.3f}\n")
        return "".join(b)

    lines.append(stats_block("mAP50 statistics", analysis["stats_map50"]))
    lines.append(stats_block("mAP50-95 statistics (PRIMARY decision metric)", analysis["stats_map50_95"]))

    s = analysis["stats_map50_95"]
    lines.append("\n## Power: seeds/config required for 80% power at the MEASURED effect size\n\n")
    if s and s["dz"] not in (0, float("inf")):
        n_req = required_n_for_power(s["dz"])
        lines.append(
            f"- measured dz (mAP50-95) = {s['dz']:.3f} -> "
            f"{'n = ' + str(n_req) if n_req else 'exceeds practical N (effect too small to reach 80% power within 1000 seeds)'}\n"
        )
    else:
        lines.append("- effect size is zero or undefined -- no finite N reaches 80% power\n")

    lines.append(
        f"\n## Verdict logic\n\n"
        "- SIGNIFICANT: primary (mAP50-95) p < 0.05 two-sided AND mean diff > 0 (proposed wins).\n"
        "- UNDERPOWERED-BUT-PROMISING: not significant, but all 5 per-seed diffs share the same "
        "sign AND |Cohen's dz| >= 0.5 (a moderate-or-larger measured effect that this n=5 study "
        "isn't powered to confirm) -- N below is the seeds/config needed for 80% power at the "
        "measured effect size.\n"
        "- PARITY: neither of the above -- either the diffs don't share a consistent sign, or the "
        "measured effect is too small to be worth chasing with more seeds.\n"
    )
    lines.append(f"\n## VERDICT: {analysis['verdict']}")
    if analysis["verdict"] == "UNDERPOWERED-BUT-PROMISING":
        lines.append(f" (N={analysis['n_required']} seeds/config needed for 80% power)")
    lines.append("\n\nNo extension, no third config, launched from this pilot. Hand-back only.\n")

    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    with open(ANALYSIS_MD, "w") as f:
        f.writelines(lines)
    print(f"wrote {ANALYSIS_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(PILOT_OUT_DIR, exist_ok=True)

    print(f"[guardrail] EPOCHS={EPOCHS} PATIENCE={PATIENCE} (matches run_step5_retrain grid values)")
    rsr.check_live_data_config()
    rsr.check_kickoff_assertion()

    if args.check_only:
        print("\n[check-only] guardrails passed. Exiting without training.")
        return 0

    device = rsr.get_device()
    print(f"[main] device: {device}")

    progress = load_progress()

    # seed 0: reused, flushed immediately, marked done
    seed0_rows = load_seed0_rows()
    for name, row in seed0_rows.items():
        entry = progress["runs"].setdefault(name, {})
        if entry.get("0") != "done":
            append_result_row(row)
            entry["0"] = "done"
            save_progress(progress)
            print(f"[seed0] reused {name}: mAP50={row['mAP50']:.4f} mAP50_95={row['mAP50_95']:.4f}")

    # seeds 1-4: train + validate through the exact grid entry point
    for config_name in CONFIGS:
        entry = progress["runs"].setdefault(config_name, {})
        for seed in SEEDS:
            if seed == 0:
                continue
            if entry.get(str(seed)) == "done":
                print(f"[{config_name} seed {seed}] already done -- skipping")
                continue
            entry[str(seed)] = "training"
            save_progress(progress)
            print(f"\n{'=' * 70}\n{config_name} seed {seed}\n{'=' * 70}")
            t0 = time.time()
            row = train_and_validate(config_name, seed, device)
            append_result_row(row)
            entry[str(seed)] = "done"
            save_progress(progress)
            print(f"[{config_name} seed {seed}] done in {time.time() - t0:.1f}s: "
                  f"mAP50={row['mAP50']:.4f} mAP50_95={row['mAP50_95']:.4f}")

    # ---- Step 2: paired analysis ----
    all_rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            all_rows.append(
                {
                    "config": r["config"],
                    "seed": int(r["seed"]),
                    "P": float(r["P"]),
                    "R": float(r["R"]),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                }
            )

    analysis = build_analysis(all_rows)
    write_analysis_md(analysis)

    print(f"\n[main] VERDICT: {analysis['verdict']}")
    print("[main] done. No extension, no third config launched -- hand-back only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
