"""
Step 3 (current task): re-validate the 10 already-trained ablation-grid
checkpoints (best.pt AND last.pt = 20 total) through the now-fixed
validation path (RGBTDetectionTrainer.get_validator() -> RGBTDetectionValidator,
wired into phase3/code/rgbt_trainer.py in Step 2).

No training runs. Each checkpoint's model object is loaded unchanged (no
get_model() rebuild/reinit) and validated once via BaseTrainer.validate(),
which is the `trainer=` code path (raw model, no AutoBackend wrapping, so
the (rgb, ir) tuple batch reaches forward() correctly) -- the same
mechanism used in diagnostics/step1b_corrected_val.py, now exercising the
REAL get_validator() instead of a diagnostics-only stub.

P/R are reported at a FIXED confidence threshold (0.25) read off the
precision/recall-vs-confidence curves so they are directly comparable
across checkpoints/configs (the metrics.box.p[0]/r[0] scalars Ultralytics
reports by default are each computed at that run's own best-F1 threshold,
which floats per checkpoint -- not a fixed point, so not used here for the
headline P/R).
"""
import gc
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from phase3.code.rgbt_trainer import RGBTDetectionTrainer

print("ultralytics version:", ultralytics.__version__)
print("torch version:", torch.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA = "phase3/dataset_aligned/data.yaml"
FIXED_CONF = 0.25

CONFIGS = [
    "abl_no_cbam_add",
    "abl_cbam_no_vqc",
    "abl_cbam_vqc_proposed",
    "abl_cbam_rgvqf_proposed",
    "abl_cbam_rgvqf_no_entangle",
    "abl_cbam_rgvqf_rem",
    "abl_cmaam_add",
    "abl_cmaam_rgvqf_rem",
    "abl_mqfp_single_scale",
    "abl_complete",
]

rows = []
t_grid0 = time.time()

for config in CONFIGS:
    for ckpt_name in ("best.pt", "last.pt"):
        weights = f"runs/detect/{config}/weights/{ckpt_name}"
        if not os.path.isfile(weights):
            print(f"SKIP missing: {weights}")
            continue

        preloaded_model = YOLO(weights).model

        class RevalTrainer(RGBTDetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                return preloaded_model

        t0 = time.time()
        trainer = RevalTrainer(overrides={
            "model": "yolo11n.pt",  # placeholder; get_model() ignores this
            "data": DATA,
            "epochs": 1,
            "imgsz": 960,
            "batch": 4,
            "device": device,
            "workers": 0,
            "conf": 0.001,  # NMS candidate threshold, NOT the reported P/R point
            "iou": 0.7,
            "plots": False,
            "project": "diagnostics/_scratch_runs",
            "name": f"revalidate_{config}_{ckpt_name.replace('.', '_')}",
            "exist_ok": True,
            "val": True,
            "rect": True,
            "box": 10.0, "cls": 0.3, "dfl": 1.5,
        })
        trainer._setup_train()
        trainer.loss_items = torch.zeros(3, device=device)
        trainer.epoch = 0
        trainer.loss = torch.zeros(1, device=device)

        metrics, fitness = trainer.validate()
        t1 = time.time()

        box = trainer.validator.metrics.box
        px = box.px
        idx = int(np.argmin(np.abs(px - FIXED_CONF)))
        p_fixed = float(box.p_curve[0][idx])
        r_fixed = float(box.r_curve[0][idx])
        map50 = float(box.map50)
        map5095 = float(box.map)

        row = dict(
            config=config, ckpt=ckpt_name,
            P=p_fixed, R=r_fixed, mAP50=map50, mAP50_95=map5095,
            wall_s=t1 - t0,
        )
        rows.append(row)
        print(f"{config:30s} {ckpt_name:9s}  P={p_fixed:.3f} R={r_fixed:.3f} "
              f"mAP50={map50:.3f} mAP50-95={map5095:.3f}  ({t1 - t0:.1f}s)")

        del trainer, preloaded_model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

t_grid1 = time.time()
print()
print(f"total wall-clock for all {len(rows)} validations: {t_grid1 - t_grid0:.1f}s")

print()
print(f"{'config':30s} {'ckpt':9s} {'P':>6s} {'R':>6s} {'mAP50':>7s} {'mAP50-95':>9s} {'wall_s':>7s}")
for r in rows:
    print(f"{r['config']:30s} {r['ckpt']:9s} {r['P']:6.3f} {r['R']:6.3f} "
          f"{r['mAP50']:7.3f} {r['mAP50_95']:9.3f} {r['wall_s']:7.1f}")

import csv
out_csv = "diagnostics/step3_revalidate_grid_results.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "ckpt", "P", "R", "mAP50", "mAP50_95", "wall_s"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out_csv}")
