#!/usr/bin/env python3
"""
Phase 1 verification: train vs val RGB/IR geometry with the patched
RGBTDetectionTrainer.build_dataset (val pad=0.0). Same checks as
audit_tmp/probe_followups.py section B, plus pass/fail asserts.
CPU, no checkpoint loaded (model built from yolo11n.pt only for stride), no training.

    .venv/bin/python -I diagnostics/presubmission/phase1_geometry_probe.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

GRAY = 114 / 255
OUT = os.path.join(ROOT, "diagnostics", "presubmission")

tr = make_trainer(ABLATIONS["cbam_add"])(overrides=dict(
    data="phase3/dataset_aligned/data.yaml", imgsz=960, batch=4, device="cpu", workers=0, rect=True,
    model="yolo11n.pt", project=os.path.join(OUT, "_scratch"), name="phase1_geometry", exist_ok=True,
    plots=False, epochs=1,
))
base = YOLO("yolo11n.pt")
tr.model = tr.get_model(cfg=base.model.yaml, weights=base.model, verbose=False)
tr.set_model_attributes()

ok = True
for mode in ("train", "val"):
    dl = tr.get_dataloader(tr.data[mode], batch_size=4, rank=-1, mode=mode)
    ds = dl.dataset
    shapes = sorted(set(map(tuple, ds.batch_shapes.tolist())))
    raw = next(iter(dl))
    if mode == "train":
        b = tr.preprocess_batch(raw)
    else:
        tr.test_loader = dl
        v = tr.get_validator()
        v.device = tr.device
        v.args.half = False
        b = v.preprocess(raw)
    rgb, ir = b["img"]
    gray_rows = sum((rgb[i, :, r, :] - GRAY).abs().max().item() < 2e-3
                    for i in range(rgb.shape[0]) for r in (0, 1, rgb.shape[2] - 2, rgb.shape[2] - 1))
    gray_cols = sum((rgb[i, :, :, c] - GRAY).abs().max().item() < 2e-3
                    for i in range(rgb.shape[0]) for c in (0, 1, rgb.shape[3] - 2, rgb.shape[3] - 1))
    rp = b.get("ratio_pad")
    print(f"{mode:5s}: dataset.pad={ds.pad} batch_shapes(unique)={shapes} rgb={list(rgb.shape)} ir={list(ir.shape)} "
          f"ratio_pad[0]={rp[0] if rp is not None else None} gray border rows/cols (of 16/16 checked)={gray_rows}/{gray_cols}")
    if mode == "val":
        checks = {
            "val tensor == [4,3,768,960]": list(rgb.shape) == [4, 3, 768, 960],
            "rgb.shape == ir.shape": rgb.shape == ir.shape,
            "batch_shapes == {(768,960)}": shapes == [(768, 960)],
            "ratio_pad padding == (0,0) for all images": rp is not None and all(tuple(p[1]) == (0, 0) or tuple(p[1]) == (0.0, 0.0) for p in rp),
            "no gray border rows/cols on RGB": gray_rows == 0 and gray_cols == 0,
            "val dataset size 1740": len(ds) == 1740,
            "val boxes 3613": sum(len(lb["cls"]) for lb in ds.labels) == 3613,
        }
        for k, v_ in checks.items():
            print(f"   [{'PASS' if v_ else 'FAIL'}] {k}")
            ok = ok and v_
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
