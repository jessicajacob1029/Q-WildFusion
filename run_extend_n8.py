#!/usr/bin/env python3
"""
Q-WildFusion n=8 extension: proposed vs baseline, adds seeds 5,6,7.

Context: the n=5 paired pilot (diagnostics/pilot_seeds/pilot_analysis.md) on
the PRIMARY metric mAP50-95 gave mean diff +0.0056, SD 0.0048, paired
t=2.59, p=0.0606, dz=1.16, 4/5 seeds favoring proposed. Underpowered, not
parity -- the pilot's own power line said n=8/config for 80% power. This run
adds seeds 5,6,7 to reach n=8 and re-tests with a fixed verdict rule (the
pilot's all-5-must-agree sign gate was too brittle and mislabeled a
p=0.06/dz=1.16 result as PARITY).

Proposed: abl_cbam_rgvqf_proposed   Baseline: abl_cbam_no_vqc
NEW seeds this run: [5, 6, 7].  Seeds 0-4 are POOLED from
diagnostics/pilot_seeds/pilot_results.csv, never retrained.

Comparability: seeds 5-7 are trained through the EXACT SAME entry point the
pilot used -- this script imports run_pilot_seeds and calls its own
train_and_validate(), which calls run_step5_retrain.train_config() /
revalidate_checkpoint() UNMODIFIED with only the module-level SEED /
RUNS_DIR / CURVES_DIR / OUT_DIR globals redirected per-run (see
run_pilot_seeds.py's module docstring). Every training hyperparameter
(EPOCHS=50, PATIENCE=20, imgsz=960, batch=4, box/cls/dfl, optimizer=auto,
fraction=1.0, rect, augmentation-off flags, deterministic=True -- Ultralytics'
default) is therefore byte-identical to seeds 0-4, so the 8 seeds are
poolable.

Guardrails:
  - No model/fusion/trainer science source is edited or reimplemented.
  - New outputs under diagnostics/pilot_seeds/ only; the n=5 files
    (pilot_results.csv, pilot_analysis.md, progress.json) are never
    overwritten -- this run writes pilot_results_n8.csv, pilot_analysis_n8.md,
    progress_n8.json.

Usage:
    caffeinate -i -s python -u run_extend_n8.py 2>&1 | tee diagnostics/pilot_seeds/run_n8.log

    python run_extend_n8.py --check-only
        Guardrail checks + resolved live data config only. No training.

Resumable: diagnostics/pilot_seeds/progress_n8.json is keyed by (config,
seed) -> pending/training/done. Re-running after a crash/reboot skips
anything already 'done'; a run left mid-training resumes via the same
Ultralytics resume=True mechanism run_step5_retrain.train_config() already
uses. Each result row is flushed (full CSV rewritten from the in-memory row
map) immediately after that run's best.pt is re-validated -- never held to
the end.
"""
import argparse
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import run_pilot_seeds as pilot  # noqa: E402  (reused entry point -- see module docstring)

rsr = pilot.rsr

# ---------------------------------------------------------------------------
# Settings -- must match the pilot exactly (do not drift)
# ---------------------------------------------------------------------------
EPOCHS = 50
PATIENCE = 20
NEW_SEEDS = [5, 6, 7]
ALL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
POOLED_SEEDS = [0, 1, 2, 3, 4]
PROPOSED_NAME = pilot.PROPOSED_NAME  # "abl_cbam_rgvqf_proposed"
BASELINE_NAME = pilot.BASELINE_NAME  # "abl_cbam_no_vqc"
CONFIGS = [PROPOSED_NAME, BASELINE_NAME]

assert EPOCHS == rsr.MODERATE_EPOCHS, (
    f"EPOCHS ({EPOCHS}) must match run_step5_retrain.MODERATE_EPOCHS ({rsr.MODERATE_EPOCHS})"
)
assert PATIENCE == rsr.PATIENCE, (
    f"PATIENCE ({PATIENCE}) must match run_step5_retrain.PATIENCE ({rsr.PATIENCE})"
)
assert EPOCHS == pilot.EPOCHS and PATIENCE == pilot.PATIENCE, (
    "EPOCHS/PATIENCE must match run_pilot_seeds too -- seeds 5-7 must be poolable with 0-4"
)

PILOT_OUT_DIR = pilot.PILOT_OUT_DIR
N5_RESULTS_CSV = pilot.RESULTS_CSV  # diagnostics/pilot_seeds/pilot_results.csv (read-only here)
N8_RESULTS_CSV = os.path.join(PILOT_OUT_DIR, "pilot_results_n8.csv")
N8_ANALYSIS_MD = os.path.join(PILOT_OUT_DIR, "pilot_analysis_n8.md")
PROGRESS_N8_PATH = os.path.join(PILOT_OUT_DIR, "progress_n8.json")

FIELDNAMES = ["config", "seed", "P", "R", "mAP50", "mAP50_95", "source"]


# ---------------------------------------------------------------------------
# Progress persistence (n8-specific -- never touches the pilot's progress.json)
# ---------------------------------------------------------------------------
def load_progress_n8():
    if os.path.isfile(PROGRESS_N8_PATH):
        with open(PROGRESS_N8_PATH) as f:
            import json

            return json.load(f)
    return {"runs": {name: {} for name in CONFIGS}}


def save_progress_n8(progress):
    import json

    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    tmp = PROGRESS_N8_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    os.replace(tmp, PROGRESS_N8_PATH)


# ---------------------------------------------------------------------------
# Result row map -> full CSV rewrite on every flush (16 rows, trivial;
# avoids append-duplication on resume)
# ---------------------------------------------------------------------------
def load_row_map():
    """Rows already known: from a prior partial n8 run if present, else empty."""
    rows = {}
    if os.path.isfile(N8_RESULTS_CSV):
        with open(N8_RESULTS_CSV) as f:
            for r in csv.DictReader(f):
                rows[(r["config"], int(r["seed"]))] = {
                    "config": r["config"],
                    "seed": int(r["seed"]),
                    "P": float(r["P"]),
                    "R": float(r["R"]),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                    "source": r["source"],
                }
    return rows


def flush_row_map(rows):
    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    tmp = N8_RESULTS_CSV + ".tmp"
    ordered = sorted(rows.values(), key=lambda x: (x["config"], x["seed"]))
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in ordered:
            w.writerow(row)
    os.replace(tmp, N8_RESULTS_CSV)


def load_pooled_seed_rows():
    """Seeds 0-4 for both configs, read from the pilot's own n=5 results CSV."""
    if not os.path.isfile(N5_RESULTS_CSV):
        raise FileNotFoundError(
            f"{N5_RESULTS_CSV} not found -- cannot pool seeds 0-4. Run the n=5 pilot first."
        )
    pooled = {}
    with open(N5_RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            if r["config"] in CONFIGS and seed in POOLED_SEEDS:
                pooled[(r["config"], seed)] = {
                    "config": r["config"],
                    "seed": seed,
                    "P": float(r["P"]),
                    "R": float(r["R"]),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                    "source": "pooled_from_pilot_n5",
                }
    expected = {(c, s) for c in CONFIGS for s in POOLED_SEEDS}
    missing = expected - set(pooled)
    if missing:
        raise ValueError(f"{N5_RESULTS_CSV} is missing rows for: {sorted(missing)}")
    return pooled


# ---------------------------------------------------------------------------
# Paired analysis over n=8, with the fixed verdict logic
# ---------------------------------------------------------------------------
def analyze(rows):
    from scipy import stats

    by = {(r["config"], r["seed"]): r for r in rows.values()}

    def diffs_for(metric):
        out = []
        for s in ALL_SEEDS:
            p = by[(PROPOSED_NAME, s)][metric]
            b = by[(BASELINE_NAME, s)][metric]
            out.append((s, p, b, p - b))
        return out

    def stats_for(metric):
        d = diffs_for(metric)
        vals = [x[3] for x in d]
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        se = sd / math.sqrt(n)
        t_stat, p_value = stats.ttest_rel(
            [x[1] for x in d], [x[2] for x in d]
        )
        df = n - 1
        tcrit = stats.t.ppf(0.975, df)
        ci = (mean - tcrit * se, mean + tcrit * se)
        dz = mean / sd if sd > 0 else float("inf")
        mean_sign = 1 if mean > 0 else (-1 if mean < 0 else 0)
        favor_count = sum(1 for v in vals if (v > 0) == (mean > 0) and v != 0)
        return {
            "per_seed": d,
            "n": n,
            "mean": mean,
            "sd": sd,
            "se": se,
            "ci": ci,
            "t": float(t_stat),
            "p": float(p_value),
            "df": df,
            "dz": dz,
            "mean_sign": mean_sign,
            "favor_count": favor_count,
        }

    s50 = stats_for("mAP50")
    s5095 = stats_for("mAP50_95")

    # FIXED verdict logic -- decided on the PRIMARY metric (mAP50-95).
    p = s5095["p"]
    mean = s5095["mean"]
    dz = abs(s5095["dz"])
    favor = s5095["favor_count"]
    n_req = pilot.required_n_for_power(s5095["dz"]) if s5095["dz"] not in (0.0, float("inf")) else None

    if p < 0.05 and mean > 0:
        verdict = "SIGNIFICANT_PROPOSED"
    elif p < 0.05 and mean < 0:
        verdict = "SIGNIFICANT_BASELINE"
    elif p >= 0.05 and dz >= 0.5 and favor >= 6:
        verdict = "PROMISING_UNDERPOWERED"
    elif p >= 0.05 and dz < 0.5:
        verdict = "PARITY"
    else:
        # p>=0.05, dz>=0.5, but <6/8 seeds share the mean's sign: large
        # apparent effect that isn't consistent across seeds -> parity.
        verdict = "PARITY"

    return {"s50": s50, "s5095": s5095, "verdict": verdict, "n_req": n_req}


def write_analysis_md(a):
    s50, s5095, verdict = a["s50"], a["s5095"], a["verdict"]
    L = []
    L.append("# n=8 extension -- paired multi-seed, proposed vs baseline\n\n")
    L.append(f"Proposed: `{PROPOSED_NAME}`  Baseline: `{BASELINE_NAME}`  Seeds: {ALL_SEEDS} (n=8)\n\n")
    L.append(
        "Seeds 0-4 pooled from `diagnostics/pilot_seeds/pilot_results.csv` (n=5 pilot, not "
        "retrained); seeds 5-7 trained here through `run_pilot_seeds.train_and_validate()` -> "
        "`run_step5_retrain.train_config()`/`revalidate_checkpoint()` unmodified. EPOCHS=50 "
        "PATIENCE=20 deterministic=True -- byte-identical to seeds 0-4.\n\n"
    )

    def block(title, s, primary=False):
        b = [f"## {title}{' (PRIMARY -- verdict decided on this)' if primary else ''}\n\n"]
        b.append("| seed | proposed | baseline | diff |\n|---|---|---|---|\n")
        for seed, pv, bv, dv in s["per_seed"]:
            b.append(f"| {seed} | {pv:.4f} | {bv:.4f} | {dv:+.4f} |\n")
        b.append(f"\n- n = {s['n']}\n")
        b.append(f"- mean diff = {s['mean']:+.4f}\n")
        b.append(f"- SD of diffs = {s['sd']:.4f}\n")
        b.append(f"- 95% CI of mean diff (t, df={s['df']}) = [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]\n")
        b.append(f"- paired t = {s['t']:.3f}, df = {s['df']}, two-sided p = {s['p']:.4f}\n")
        b.append(f"- Cohen's dz = {s['dz']:.3f}\n")
        b.append(f"- sign consistency = {s['favor_count']}/8 seeds favor the mean's direction "
                 f"({'proposed' if s['mean'] > 0 else 'baseline'})\n\n")
        return "".join(b)

    L.append(block("mAP50-95 statistics", s5095, primary=True))
    L.append(block("mAP50 statistics", s50))

    L.append("## Power at the n=8 measured effect\n\n")
    if a["n_req"] is not None:
        L.append(f"- dz (mAP50-95) = {s5095['dz']:.3f} -> {a['n_req']} seeds/config for 80% power "
                 f"(two-sided alpha=0.05)\n\n")
    else:
        L.append("- effect size zero/undefined -- no finite N reaches 80% power\n\n")

    L.append(
        "## Verdict logic (fixed -- primary metric mAP50-95)\n\n"
        "- SIGNIFICANT_PROPOSED : p < 0.05 AND mean diff > 0\n"
        "- SIGNIFICANT_BASELINE : p < 0.05 AND mean diff < 0\n"
        "- PROMISING_UNDERPOWERED : p >= 0.05 AND |dz| >= 0.5 AND >= 6/8 seeds share the mean's sign\n"
        "- PARITY : p >= 0.05 AND |dz| < 0.5  (also: p>=0.05 with large dz but <6/8 sign agreement)\n\n"
    )

    L.append(f"## VERDICT (mAP50-95): {verdict}\n\n")
    L.append(
        f"- p = {s5095['p']:.4f}, mean diff = {s5095['mean']:+.4f}, |dz| = {abs(s5095['dz']):.3f}, "
        f"sign agreement = {s5095['favor_count']}/8\n"
    )
    if verdict == "PROMISING_UNDERPOWERED" and a["n_req"] is not None:
        L.append(f"- N for 80% power at the measured effect: {a['n_req']} seeds/config\n")
    L.append(
        f"\n## mAP50 (secondary -- stated separately so the metrics aren't conflated)\n\n"
        f"- p = {s50['p']:.4f}, mean diff = {s50['mean']:+.4f}, |dz| = {abs(s50['dz']):.3f}, "
        f"sign agreement = {s50['favor_count']}/8. "
        f"{'Parity on mAP50.' if s50['p'] >= 0.05 else 'Significant on mAP50 -- investigate.'}\n"
    )
    L.append("\nHand-back only. No further seeds, no third config launched from this run.\n")

    os.makedirs(PILOT_OUT_DIR, exist_ok=True)
    with open(N8_ANALYSIS_MD, "w") as f:
        f.writelines(L)
    print(f"wrote {N8_ANALYSIS_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(PILOT_OUT_DIR, exist_ok=True)

    print(f"[guardrail] EPOCHS={EPOCHS} PATIENCE={PATIENCE} "
          f"(matches run_step5_retrain AND run_pilot_seeds)")
    rsr.check_live_data_config()
    rsr.check_kickoff_assertion()

    if args.check_only:
        print("\n[check-only] guardrails passed. Exiting without training.")
        return 0

    device = rsr.get_device()
    print(f"[main] device: {device}")

    progress = load_progress_n8()
    rows = load_row_map()

    # pool seeds 0-4 from the pilot's CSV, flush immediately
    pooled = load_pooled_seed_rows()
    for kseed, row in pooled.items():
        if kseed not in rows:
            rows[kseed] = row
    flush_row_map(rows)
    for name in CONFIGS:
        for s in POOLED_SEEDS:
            progress["runs"].setdefault(name, {})[str(s)] = "done"
    save_progress_n8(progress)
    print(f"[pool] seeds 0-4 for both configs loaded from {os.path.basename(N5_RESULTS_CSV)}")

    # train seeds 5,6,7 for both configs
    for config_name in CONFIGS:
        entry = progress["runs"].setdefault(config_name, {})
        for seed in NEW_SEEDS:
            if entry.get(str(seed)) == "done" and (config_name, seed) in rows:
                print(f"[{config_name} seed {seed}] already done -- skipping")
                continue
            entry[str(seed)] = "training"
            save_progress_n8(progress)
            print(f"\n{'=' * 70}\n{config_name} seed {seed}\n{'=' * 70}")
            import time

            t0 = time.time()
            row = pilot.train_and_validate(config_name, seed, device)
            row["source"] = "trained_by_n8"
            rows[(config_name, seed)] = row
            flush_row_map(rows)
            entry[str(seed)] = "done"
            save_progress_n8(progress)
            print(f"[{config_name} seed {seed}] done in {time.time() - t0:.1f}s: "
                  f"mAP50={row['mAP50']:.4f} mAP50_95={row['mAP50_95']:.4f}")

    # ---- Step 2: paired analysis over all 8 seeds ----
    a = analyze(rows)
    write_analysis_md(a)

    print(f"\n[main] VERDICT (mAP50-95): {a['verdict']}")
    print(f"[main]   p={a['s5095']['p']:.4f} mean_diff={a['s5095']['mean']:+.4f} "
          f"|dz|={abs(a['s5095']['dz']):.3f} sign={a['s5095']['favor_count']}/8")
    print(f"[main]   mAP50 (secondary): p={a['s50']['p']:.4f} mean_diff={a['s50']['mean']:+.4f}")
    print("[main] done. Hand-back only -- no further seeds, no third config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
