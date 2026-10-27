#!/usr/bin/env python3
"""
Q-WildFusion Step 5 -- staged ablation-grid retrain with the corrected
in-loop validator.

Context: Task A (diagnostics/step4b_verification.md) verified
RGBTDetectionTrainer.get_validator() installs the corrected
RGBTDetectionValidator as the trainer-time / checkpoint-selection /
patience validator. Task B verified phase3/dataset_aligned/data.yaml is
the live, correct data root. Step 0 (diagnostics/step5_retrain/step0_report.md)
measured real per-epoch cost and found the full 10-config/50-epoch grid is a
genuinely multi-day job on this machine (~216.5h at no-early-stop-credit
ceiling) -- far past the original 36h budget knob. Running this script is
the operator's explicit acceptance of that cost.

This script does NOT edit any model/fusion/trainer science source. It only
imports and drives the existing, unmodified RGBTDetectionTrainer /
RGBTDetectionValidator / train_ablation_study.ABLATIONS. All outputs go
under diagnostics/step5_retrain/ (progress.json, curves/, ranked_grid.csv/.md,
step2_report.md). Training runs land in a NEW directory
(diagnostics/step5_retrain/runs/) -- the original Step 3/4 artifacts under
runs/detect/abl_* are never touched or overwritten.

Usage:
    caffeinate -i -s python -u run_step5_retrain.py 2>&1 | tee diagnostics/step5_retrain/run.log

    python run_step5_retrain.py --check-only
        Runs only the guardrail checks (live-data-config resolution, kickoff
        assertion) and the config->module audit print. No training, no
        validation, no filesystem writes under runs/. Exits 0 on success.

Resumable: diagnostics/step5_retrain/progress.json is written after every
config's train step and every validate step. Re-running this script after
an interruption (crash, sleep, session/harness restart -- this has happened
before in this project) picks up where it left off:
  - a config whose last.pt has epoch == -1 (Ultralytics' own "cleanly
    finished" sentinel, set by strip_optimizer() inside final_eval() -- see
    site-packages/ultralytics/utils/torch_utils.py::strip_optimizer and
    engine/trainer.py::resume_training's identical check) is skipped;
  - a config with a last.pt that has NOT been finalized (mid-training) is
    resumed via Ultralytics' own model.train(resume=True);
  - a config with no run directory yet is trained fresh;
  - already-computed re-validation rows are skipped on rerun.
"""
import argparse
import inspect
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Budget knobs -- edit before running
# ---------------------------------------------------------------------------
MAX_GRID_HOURS = 36
# Informational only at this point in the pipeline: Step 0's cost gate
# already ran (diagnostics/step5_retrain/step0_report.md) and FAILED at this
# budget for the full 50-epoch/10-config grid (~216.5h extrapolated, no
# early-stop credit). This script does not re-block on MAX_GRID_HOURS --
# choosing to launch it at all is the acceptance of the real, multi-day
# cost. It's kept here only so the elapsed-vs-budget line printed at the end
# of the run means something.
MODERATE_EPOCHS = 50
PATIENCE = 20
SEED = 0

DATA_YAML = "phase3/dataset_aligned/data.yaml"
FIXED_CONF = 0.25  # matches the Step 3 re-validation convention

OUT_DIR = os.path.join(ROOT, "diagnostics", "step5_retrain")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
CURVES_DIR = os.path.join(OUT_DIR, "curves")
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")

ONE_FIRST_KEY = "cbam_add"  # train_ablation_study.ABLATIONS key whose "name" is "abl_cbam_no_vqc"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
def check_live_data_config():
    """Confirm which data.yaml is actually live (has the ir: key), matching
    what rgbt_trainer.py / train.py / train_ablation_study.py load. Never
    the dead top-level phase3/data.yaml or ./data.yaml (missing ir:,
    KeyError at rgbt_trainer.py:127 -- diagnostics/step4b_verification.md)."""
    data_path = os.path.join(ROOT, DATA_YAML)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Live data config not found: {data_path}")
    with open(data_path) as f:
        content = f.read()
    if "ir:" not in content:
        raise AssertionError(f"{DATA_YAML} is missing the 'ir:' key -- this is not the live config. STOP.")
    print(f"[guardrail] live data config resolves to: {DATA_YAML}")
    print(content)
    return data_path


def check_kickoff_assertion():
    """Cheap assertion (not a re-probe): get_validator() must reference
    RGBTDetectionValidator, per Task A's verified fix."""
    from phase3.code.rgbt_trainer import RGBTDetectionTrainer, RGBTDetectionValidator

    src = inspect.getsource(RGBTDetectionTrainer.get_validator)
    assert "RGBTDetectionValidator" in src, (
        "get_validator() does not reference RGBTDetectionValidator -- the corrected "
        "validator is not wired in. STOP, do not train."
    )
    print("[guardrail] RGBTDetectionTrainer.get_validator() source:")
    print(src)
    print(f"[guardrail] resolves to class: {RGBTDetectionValidator}")
    return True


def print_config_audit():
    """List the 10 configs with their exact module settings, reused
    verbatim from train_ablation_study.ABLATIONS, so the config->module
    mapping is auditable before anything launches."""
    from phase3.code.train_ablation_study import ABLATIONS

    print("=" * 78)
    print("CONFIG -> MODULE AUDIT (phase3/code/train_ablation_study.ABLATIONS, unmodified)")
    print("=" * 78)
    for key, cfg in ABLATIONS.items():
        marker = " <- ONE-FIRST" if key == ONE_FIRST_KEY else ""
        print(
            f"  {key:30s} -> {cfg['name']:28s} fusion={cfg['fusion_mode']:8s} "
            f"cbam={cfg.get('use_cbam', True)!s:5s} attention={str(cfg.get('attention')):6s} "
            f"entangle={cfg.get('use_entanglement', True)!s:5s} "
            f"reliability={cfg.get('use_reliability', False)!s:5s} "
            f"multiscale={cfg.get('multiscale', True)!s:5s} "
            f"alpha_rel={cfg.get('alpha_rel', 0.0)} beta_q={cfg.get('beta_q', 0.0)}{marker}"
        )
    assert ONE_FIRST_KEY in ABLATIONS, f"ONE_FIRST_KEY {ONE_FIRST_KEY!r} not in ABLATIONS"
    return ABLATIONS


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------
def load_progress():
    if os.path.isfile(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"one_first": {"done": False, "passed": False, "detail": None}, "configs": {}}


def save_progress(progress):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    os.replace(tmp, PROGRESS_PATH)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Training (with resume)
# ---------------------------------------------------------------------------
def _run_is_finalized(last_pt):
    """True iff last_pt exists and carries Ultralytics' own "cleanly
    finished" sentinel (epoch == -1, written by strip_optimizer() inside
    final_eval(), which only runs once the training loop has broken --
    either via patience or by reaching the epoch cap). Verified against
    the installed ultralytics source, not assumed:
      - utils/torch_utils.py::strip_optimizer sets x["epoch"] = -1
      - engine/trainer.py::resume_training does
            start_epoch = ckpt.get("epoch", -1) + 1
            assert start_epoch > 0, "... training ... is finished, nothing to resume."
        i.e. Ultralytics itself uses this exact sentinel to refuse resuming
        a finished run."""
    if not os.path.isfile(last_pt):
        return False
    import torch

    try:
        ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(ckpt, dict) and ckpt.get("epoch", None) == -1


def train_config(key, config, device):
    from ultralytics import YOLO

    from phase3.code.train_ablation_study import make_trainer

    run_dir = os.path.join(RUNS_DIR, config["name"])
    weights_dir = os.path.join(run_dir, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    last_pt = os.path.join(weights_dir, "last.pt")
    results_csv = os.path.join(run_dir, "results.csv")

    if _run_is_finalized(last_pt):
        print(f"[{key}] already trained (last.pt carries the finished sentinel) -- skipping training")
    else:
        t0 = time.time()
        if os.path.isfile(last_pt):
            print(f"[{key}] found partial run at {last_pt} -- resuming via Ultralytics resume=True")
            model = YOLO(last_pt)
            model.train(resume=True)
        else:
            print(f"[{key}] fresh training -> {run_dir}")
            trainer_cls = make_trainer(config)
            model = YOLO("yolo11n.pt")
            model.train(
                trainer=trainer_cls,
                data=DATA_YAML,
                epochs=MODERATE_EPOCHS,
                imgsz=960,
                batch=4,
                device=device,
                workers=0,
                project=RUNS_DIR,
                name=config["name"],
                exist_ok=True,
                fraction=1.0,
                patience=PATIENCE,
                seed=SEED,
                pretrained=True,
                box=10.0,
                cls=0.3,
                dfl=1.5,
                rect=True,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.0,
                close_mosaic=0,
                val=True,
                plots=False,
            )
        t1 = time.time()
        print(f"[{key}] training finished in {t1 - t0:.1f}s")

    if os.path.isfile(results_csv):
        os.makedirs(CURVES_DIR, exist_ok=True)
        dest = os.path.join(CURVES_DIR, f"{config['name']}.csv")
        shutil.copy(results_csv, dest)
        print(f"[{key}] val-metric-vs-epoch curve -> {dest}")

    assert os.path.isfile(best_pt) and os.path.isfile(last_pt), (
        f"[{key}] training reported done but weights are missing at {weights_dir}"
    )
    return run_dir, best_pt, last_pt


# ---------------------------------------------------------------------------
# Re-validation (same mechanism as Step 3: trainer-branch validate(), no
# AutoBackend, corrected RGBTDetectionValidator, P/R read at FIXED_CONF)
# ---------------------------------------------------------------------------
def revalidate_checkpoint(weights_path, device):
    import gc

    import numpy as np
    import torch
    from ultralytics import YOLO

    from phase3.code.rgbt_trainer import RGBTDetectionTrainer

    preloaded_model = YOLO(weights_path).model

    class RevalTrainer(RGBTDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return preloaded_model

    trainer = RevalTrainer(
        overrides={
            "model": "yolo11n.pt",
            "data": DATA_YAML,
            "epochs": 1,
            "imgsz": 960,
            "batch": 4,
            "device": device,
            "workers": 0,
            "conf": 0.001,
            "iou": 0.7,
            "plots": False,
            "project": os.path.join(OUT_DIR, "_scratch_runs"),
            "name": "reval_" + os.path.basename(os.path.dirname(os.path.dirname(weights_path))),
            "exist_ok": True,
            "val": True,
            "rect": True,
            "box": 10.0,
            "cls": 0.3,
            "dfl": 1.5,
        }
    )
    trainer._setup_train()
    trainer.loss_items = torch.zeros(3, device=device)
    trainer.epoch = 0
    trainer.loss = torch.zeros(1, device=device)

    metrics, fitness = trainer.validate()

    box = trainer.validator.metrics.box
    px = box.px
    idx = int(np.argmin(np.abs(px - FIXED_CONF)))
    result = {
        "P": float(box.p_curve[0][idx]),
        "R": float(box.r_curve[0][idx]),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "fitness": float(fitness) if fitness is not None else None,
    }

    del trainer, preloaded_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# One-first sanity checks
# ---------------------------------------------------------------------------
def read_fitness_curve(results_csv):
    """Return list of (epoch, fitness) from Ultralytics' own results.csv.
    Ultralytics writes a 'fitness' column directly; fall back to the
    standard 0.1*mAP50 + 0.9*mAP50-95 weighting from metrics/mAP50(B) and
    metrics/mAP50-95(B) if it's absent in this ultralytics version."""
    import csv as _csv

    rows = []
    with open(results_csv) as f:
        reader = _csv.DictReader(f)
        for r in reader:
            r = {k.strip(): v for k, v in r.items()}
            rows.append(r)
    curve = []
    for r in rows:
        epoch = int(float(r.get("epoch", len(curve) + 1)))
        if "fitness" in r and r["fitness"] not in ("", None):
            fit = float(r["fitness"])
        else:
            map50 = float(r.get("metrics/mAP50(B)", 0.0))
            map5095 = float(r.get("metrics/mAP50-95(B)", 0.0))
            fit = 0.1 * map50 + 0.9 * map5095
        curve.append((epoch, fit))
    return curve


def one_first_sanity_check(config, run_dir, best_pt, last_pt, device):
    """Three checks, per the task spec:
      1. best.pt fitness >= last.pt fitness (the historic inversion is gone).
      2. patience fired on a real plateau, not early noise: (a) if the run
         early-stopped, the best epoch must be PATIENCE full epochs before
         the last completed epoch, matching Ultralytics' own EarlyStopping
         definition; (b) the best epoch isn't in the first few epochs
         (pure epoch-1-3 noise) unless the run's total length made that
         unavoidable; (c) fitness at the best epoch is meaningfully above
         fitness ~5 epochs earlier -- there was a real climb into it, not a
         flat/noisy plateau the whole way.
      3. P/R in a sane range on best.pt: R not collapsed, P not pinned at 1.0
         -- the exact broken signature this whole investigation started from.
    Returns (passed: bool, detail: dict).
    """
    detail = {}

    best_metrics = revalidate_checkpoint(best_pt, device)
    last_metrics = revalidate_checkpoint(last_pt, device)
    detail["best_metrics"] = best_metrics
    detail["last_metrics"] = last_metrics

    check1 = best_metrics["fitness"] is not None and last_metrics["fitness"] is not None and (
        best_metrics["fitness"] >= last_metrics["fitness"] - 1e-6
    )
    detail["check1_best_ge_last_fitness"] = check1

    results_csv = os.path.join(run_dir, "results.csv")
    curve = read_fitness_curve(results_csv)
    epochs_run = len(curve)
    fits = [f for _, f in curve]
    best_idx = max(range(len(fits)), key=lambda i: fits[i])
    best_epoch = curve[best_idx][0]

    early_stopped = epochs_run < MODERATE_EPOCHS
    plateau_gap_ok = (not early_stopped) or (epochs_run - best_idx - 1 >= PATIENCE - 1)
    not_pure_early_noise = best_idx >= 2 or epochs_run <= 3
    lookback = max(0, best_idx - 5)
    real_climb = fits[best_idx] > fits[lookback] + 1e-3 if best_idx > lookback else True
    check2 = plateau_gap_ok and not_pure_early_noise and real_climb
    detail["check2_real_plateau"] = {
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "best_idx": best_idx,
        "plateau_gap_ok": plateau_gap_ok,
        "not_pure_early_noise": not_pure_early_noise,
        "real_climb": real_climb,
        "passed": check2,
    }

    check3 = best_metrics["R"] > 0.15 and best_metrics["P"] < 0.99
    detail["check3_sane_pr"] = {"R": best_metrics["R"], "P": best_metrics["P"], "passed": check3}

    passed = bool(check1 and check2 and check3)
    detail["passed"] = passed
    return passed, detail


# ---------------------------------------------------------------------------
# Step 2: ranked report, flag-don't-act
# ---------------------------------------------------------------------------
NAMED_COMPARISONS = [
    # (label, config_name_A, config_name_B) -- decision-critical pairs the
    # task asked to check against each config's own best<->last spread.
    ("proposed (cbam+rgvqf) vs baseline (cbam+add)", "abl_cbam_rgvqf_proposed", "abl_cbam_no_vqc"),
    ("entangle vs no-entangle (rgvqf)", "abl_cbam_rgvqf_proposed", "abl_cbam_rgvqf_no_entangle"),
    ("complete vs proposed", "abl_complete", "abl_cbam_rgvqf_proposed"),
]


def build_ranked_report(all_results, progress):
    """all_results: {config_name: {"best": {...}, "last": {...}, "epochs_run": int}}"""
    rows = []
    for name, r in all_results.items():
        rows.append(
            {
                "config": name,
                "checkpoint": "best.pt",
                "P": r["best"]["P"],
                "R": r["best"]["R"],
                "mAP50": r["best"]["mAP50"],
                "mAP50_95": r["best"]["mAP50_95"],
            }
        )
        rows.append(
            {
                "config": name,
                "checkpoint": "last.pt",
                "P": r["last"]["P"],
                "R": r["last"]["R"],
                "mAP50": r["last"]["mAP50"],
                "mAP50_95": r["last"]["mAP50_95"],
            }
        )
    rows.sort(key=lambda r: r["mAP50_95"], reverse=True)

    csv_path = os.path.join(OUT_DIR, "ranked_grid.csv")
    with open(csv_path, "w", newline="") as f:
        import csv as _csv

        w = _csv.DictWriter(f, fieldnames=["config", "checkpoint", "P", "R", "mAP50", "mAP50_95"])
        w.writeheader()
        w.writerows(rows)

    # (a) still climbing at the cap
    still_climbing = []
    for name, r in all_results.items():
        curve_path = os.path.join(CURVES_DIR, f"{name}.csv")
        if not os.path.isfile(curve_path):
            continue
        curve = read_fitness_curve(curve_path)
        epochs_run = len(curve)
        if epochs_run < MODERATE_EPOCHS:
            continue  # early-stopped, not "still climbing at the cap"
        fits = [f for _, f in curve]
        lookback = max(0, len(fits) - 6)
        if fits[-1] > fits[lookback] + 1e-3:
            still_climbing.append(name)

    # (b) decision-critical gaps smaller than either config's own best<->last spread
    noisy_comparisons = []
    for label, name_a, name_b in NAMED_COMPARISONS:
        if name_a not in all_results or name_b not in all_results:
            continue
        a, b = all_results[name_a], all_results[name_b]
        gap = abs(a["best"]["mAP50_95"] - b["best"]["mAP50_95"])
        spread_a = abs(a["best"]["mAP50_95"] - a["last"]["mAP50_95"])
        spread_b = abs(b["best"]["mAP50_95"] - b["last"]["mAP50_95"])
        noise_floor = max(spread_a, spread_b)
        if gap < noise_floor:
            noisy_comparisons.append(
                {"label": label, "a": name_a, "b": name_b, "gap": gap, "noise_floor": noise_floor}
            )

    md_path = os.path.join(OUT_DIR, "ranked_grid.md")
    with open(md_path, "w") as f:
        f.write("# Step 5 retrain -- ranked grid (mAP50-95 descending)\n\n")
        f.write(f"MODERATE_EPOCHS={MODERATE_EPOCHS}  PATIENCE={PATIENCE}  SEED={SEED}\n\n")
        f.write("| config | checkpoint | P | R | mAP50 | mAP50-95 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['config']} | {r['checkpoint']} | {r['P']:.3f} | {r['R']:.3f} | "
                f"{r['mAP50']:.3f} | {r['mAP50_95']:.3f} |\n"
            )
        f.write("\n## Flagged -- not acted on, hand back only\n\n")
        f.write("### (a) still climbing at the epoch cap (undertrained, extension candidate)\n\n")
        if still_climbing:
            for name in still_climbing:
                f.write(f"- {name}\n")
        else:
            f.write("- none\n")
        f.write(
            "\n### (b) decision-critical gaps smaller than the configs' own best<->last spread "
            "(within noise, needs seeds/extension before it can be claimed)\n\n"
        )
        if noisy_comparisons:
            for nc in noisy_comparisons:
                f.write(
                    f"- **{nc['label']}** ({nc['a']} vs {nc['b']}): "
                    f"gap={nc['gap']:.4f} < noise floor={nc['noise_floor']:.4f}\n"
                )
        else:
            f.write("- none\n")
        f.write(
            "\nNo extensions or multi-seed runs were launched. This is a hand-back for a decision, "
            "not an auto-continuation.\n"
        )

    print(f"[step2] wrote {csv_path}")
    print(f"[step2] wrote {md_path}")
    return csv_path, md_path, still_climbing, noisy_comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run guardrail checks + config audit only. No training, no validation, no writes under runs/.",
    )
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    check_live_data_config()
    check_kickoff_assertion()
    ablations = print_config_audit()

    if args.check_only:
        print("\n[check-only] guardrails + audit passed. Exiting without training.")
        return 0

    device = get_device()
    print(f"\n[main] device: {device}")

    progress = load_progress()

    # ---- one-first ----
    if not progress["one_first"]["done"]:
        config = ablations[ONE_FIRST_KEY]
        print(f"\n{'=' * 78}\nONE-FIRST: {config['name']}\n{'=' * 78}")
        run_dir, best_pt, last_pt = train_config(ONE_FIRST_KEY, config, device)
        passed, detail = one_first_sanity_check(config, run_dir, best_pt, last_pt, device)
        progress["one_first"] = {"done": True, "passed": passed, "detail": detail}
        save_progress(progress)
        print(f"\n[one-first] PASSED={passed}")
        print(json.dumps(detail, indent=2, default=str))
        if not passed:
            print("\n[one-first] FAILED -- STOPPING. The other nine configs were NOT launched.")
            return 1
    else:
        print(f"\n[one-first] already done (passed={progress['one_first']['passed']}) -- skipping")
        if not progress["one_first"]["passed"]:
            print("[one-first] progress.json shows a prior FAIL -- STOPPING without re-running.")
            return 1

    # ---- remaining nine ----
    remaining_keys = [k for k in ablations if k != ONE_FIRST_KEY]
    for key in remaining_keys:
        config = ablations[key]
        cfg_progress = progress["configs"].setdefault(key, {"trained": False, "validated": False})
        print(f"\n{'=' * 78}\n{key} -> {config['name']}\n{'=' * 78}")
        run_dir, best_pt, last_pt = train_config(key, config, device)
        cfg_progress["trained"] = True
        cfg_progress["run_dir"] = run_dir
        save_progress(progress)

        if not cfg_progress.get("validated"):
            best_metrics = revalidate_checkpoint(best_pt, device)
            last_metrics = revalidate_checkpoint(last_pt, device)
            curve = read_fitness_curve(os.path.join(run_dir, "results.csv"))
            cfg_progress["best_metrics"] = best_metrics
            cfg_progress["last_metrics"] = last_metrics
            cfg_progress["epochs_run"] = len(curve)
            cfg_progress["validated"] = True
            save_progress(progress)
            print(f"[{key}] best: {best_metrics}")
            print(f"[{key}] last: {last_metrics}")

    # also validate the one-first config's final numbers into the same shape, if not already present
    one_first_cfg = ablations[ONE_FIRST_KEY]
    of_progress = progress["configs"].setdefault(ONE_FIRST_KEY, {})
    if "best_metrics" not in of_progress:
        of_detail = progress["one_first"]["detail"]
        of_progress["best_metrics"] = of_detail["best_metrics"]
        of_progress["last_metrics"] = of_detail["last_metrics"]
        curve = read_fitness_curve(os.path.join(RUNS_DIR, one_first_cfg["name"], "results.csv"))
        of_progress["epochs_run"] = len(curve)
        of_progress["validated"] = True
        save_progress(progress)

    # ---- Step 2: ranked report ----
    all_results = {}
    for key, cfg in ablations.items():
        cp = progress["configs"].get(key)
        if not cp or "best_metrics" not in cp:
            print(f"[step2] WARNING: no validated metrics for {key} ({cfg['name']}) -- skipping in report")
            continue
        all_results[cfg["name"]] = {
            "best": cp["best_metrics"],
            "last": cp["last_metrics"],
            "epochs_run": cp.get("epochs_run"),
        }

    build_ranked_report(all_results, progress)

    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n[main] total elapsed: {elapsed_h:.2f}h (budget knob MAX_GRID_HOURS={MAX_GRID_HOURS}h)")
    print("[main] done. Ranked table + flags are in diagnostics/step5_retrain/ranked_grid.{csv,md}")
    print("[main] no extensions or multi-seed runs were launched -- hand-back only, per the task spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
