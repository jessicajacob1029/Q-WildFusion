"""
Step 0b: one real, FULL epoch (imgsz=960, batch=4 -- the actual grid
settings, not the CLI defaults in train_ablation_study.py) on one fast
config (no_cbam_add) and one quantum config (cbam_rgvqf), through the
corrected validator, to measure real per-epoch TRAIN and VAL wall-clock.

Reuses train_ablation_study.ABLATIONS / make_trainer directly so the probe
exercises the exact same config->module wiring the real grid will use.
No science files edited. Outputs under diagnostics/step5_retrain/ only.
"""
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import ultralytics
from ultralytics import YOLO

from phase3.code.train_ablation_study import ABLATIONS, make_trainer

print("ultralytics version:", ultralytics.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA = "phase3/dataset_aligned/data.yaml"
PROBE_CONFIGS = ["no_cbam_add", "cbam_rgvqf"]

results = {}

for key in PROBE_CONFIGS:
    config = ABLATIONS[key]
    print()
    print("=" * 70)
    print(f"PROBE: {key} -> {config['name']}  (fusion_mode={config['fusion_mode']})")
    print("=" * 70)

    trainer_cls = make_trainer(config)

    timings = {"train_epoch_start": None, "train_epoch_end": None, "val_start": None, "val_end": None}

    def _on_train_epoch_start(trainer):
        timings["train_epoch_start"] = time.time()

    def _on_train_epoch_end(trainer):
        timings["train_epoch_end"] = time.time()

    def _on_val_start(validator):
        timings["val_start"] = time.time()

    def _on_val_end(validator):
        timings["val_end"] = time.time()

    model = YOLO("yolo11n.pt")
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_val_start", _on_val_start)
    model.add_callback("on_val_end", _on_val_end)

    t0 = time.time()
    model.train(
        trainer=trainer_cls,
        data=DATA,
        epochs=1,
        imgsz=960,
        batch=4,
        device=device,
        workers=0,
        project="diagnostics/step5_retrain/timing_probe",
        name=config["name"],
        exist_ok=True,
        fraction=1.0,
        patience=999,
        pretrained=True,
        box=10.0,
        cls=0.3,
        dfl=1.5,
        rect=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0,
        val=True,
        plots=False,
    )
    t1 = time.time()

    train_s = timings["train_epoch_end"] - timings["train_epoch_start"] if timings["train_epoch_end"] else None
    val_s = timings["val_end"] - timings["val_start"] if timings["val_end"] else None

    results[key] = {
        "config_name": config["name"],
        "fusion_mode": config["fusion_mode"],
        "total_wall_s": t1 - t0,
        "train_epoch_s": train_s,
        "val_s": val_s,
    }
    print(f"RESULT {key}: total={t1 - t0:.1f}s  train_epoch={train_s:.1f}s  val={val_s:.1f}s")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for key, r in results.items():
    print(f"{key:20s} ({r['config_name']:28s} fusion={r['fusion_mode']:6s}): "
          f"train_epoch={r['train_epoch_s']:.1f}s  val={r['val_s']:.1f}s  total={r['total_wall_s']:.1f}s")

fast = results["no_cbam_add"]
quantum = results["cbam_rgvqf"]
ratio_train = quantum["train_epoch_s"] / fast["train_epoch_s"]
ratio_total = quantum["total_wall_s"] / fast["total_wall_s"]
print()
print(f"fast:quantum train-epoch ratio = 1 : {ratio_train:.2f}")
print(f"fast:quantum total (train+val) ratio = 1 : {ratio_total:.2f}")

import json
with open("diagnostics/step5_retrain/step0b_timing_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote diagnostics/step5_retrain/step0b_timing_results.json")
