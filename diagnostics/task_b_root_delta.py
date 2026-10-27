"""
Task B step 4: validate ONE representative checkpoint (abl_no_cbam_add/best.pt,
already re-validated in Step 3 against dataset_aligned) through the corrected
path against BOTH candidate dataset roots, to quantify how much the root choice
moves the metrics. No training. No file under phase3/code is modified.

Root 1: phase3/data.yaml (path: phase3/dataset) -- the top-level file, unused
         by any active script in the repo (confirmed by grep), path: dataset
         (not aligned), missing an 'ir:'/'thermal:' key entirely.
Root 2: phase3/dataset_aligned/data.yaml -- confirmed used by every training/
         eval/gradcam/diagnostic script in the repo.
"""
import csv
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import ultralytics
from ultralytics import YOLO

from phase3.code.rgbt_trainer import RGBTDetectionTrainer

print("ultralytics version:", ultralytics.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

WEIGHTS = "runs/detect/abl_no_cbam_add/weights/best.pt"
ROOTS = {
    "phase3/data.yaml (path: dataset)": "phase3/data.yaml",
    "phase3/dataset_aligned/data.yaml": "phase3/dataset_aligned/data.yaml",
}

rows = []

for label, data_yaml in ROOTS.items():
    print()
    print("=" * 70)
    print(f"ROOT: {label}  ({data_yaml})")
    print("=" * 70)

    preloaded_model = YOLO(WEIGHTS).model

    class RevalTrainer(RGBTDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return preloaded_model

    t0 = time.time()
    try:
        trainer = RevalTrainer(overrides={
            "model": "yolo11n.pt",
            "data": data_yaml,
            "epochs": 1,
            "imgsz": 960,
            "batch": 4,
            "device": device,
            "workers": 0,
            "conf": 0.001,
            "iou": 0.7,
            "plots": False,
            "project": "diagnostics/_scratch_runs",
            "name": f"task_b_{label.split()[0].replace('/', '_')}",
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
        map50 = float(box.map50)
        map5095 = float(box.map)
        print(f"SUCCESS  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  wall={t1 - t0:.1f}s")
        rows.append(dict(root=label, data_yaml=data_yaml, status="SUCCESS",
                          mAP50=map50, mAP50_95=map5095, wall_s=t1 - t0, error=""))
    except Exception as e:
        t1 = time.time()
        import traceback
        tb = traceback.format_exc()
        print(f"FAILED after {t1 - t0:.1f}s")
        print(tb)
        rows.append(dict(root=label, data_yaml=data_yaml, status="FAILED",
                          mAP50=None, mAP50_95=None, wall_s=t1 - t0, error=str(e)))

    del preloaded_model
    import gc
    gc.collect()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for r in rows:
    print(r)

out_csv = "diagnostics/task_b_root_delta.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["root", "data_yaml", "status", "mAP50", "mAP50_95", "wall_s", "error"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out_csv}")
