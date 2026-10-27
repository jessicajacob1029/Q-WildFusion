#!/usr/bin/env python3
"""
Phase 2 cross-check: is the large metric jump caused by RGB/IR registration,
or by the pad change itself?

Validates one checkpoint three ways, all through the revalidate_checkpoint
recipe (trainer.validate(), conf .001, iou .7, imgsz 960, rect, P/R@0.25):
  A  old geometry : val pad=0.5 (800x992), IR stretched           (pre-fix behaviour)
  B  aligned pad  : val pad=0.5 (800x992), IR LETTERBOXED like RGB (same 16px gray border)
  C  fixed        : val pad=0.0 (768x960), IR resized             (Phase 1 fix)
If registration is the cause: B ~= C >> A.
Probe-only subclasses; repo code untouched. Checkpoint read-only.

    .venv/bin/python -u diagnostics/presubmission/phase2_crosscheck.py [weights]
"""
import gc
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

import run_step5_retrain as rsr  # noqa: E402
from phase3.code.rgbt_trainer import RGBTBatch, RGBTDetectionTrainer, RGBTDetectionValidator  # noqa: E402

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/step5_retrain/runs/abl_no_cbam_add/weights/best.pt"
OUT = os.path.join(ROOT, "diagnostics", "presubmission")


class LetterboxIRValidator(RGBTDetectionValidator):
    """IR letterboxed with the RGB batch's own ratio_pad and the same 114 gray border."""

    def preprocess(self, batch):
        batch = super(RGBTDetectionValidator, self).preprocess(batch)  # stock device/normalise
        rgb = batch["img"]
        H, W = rgb.shape[-2:]
        ir_root = self._rgbt_trainer._resolve_ir_root()
        irs = []
        for rgb_path, (ratio, (padw, padh)) in zip(batch["im_file"], batch["ratio_pad"]):
            ir = cv2.imread(self._rgbt_trainer._matching_ir_path(str(rgb_path), ir_root), cv2.IMREAD_GRAYSCALE)
            h0, w0 = ir.shape
            nw, nh = int(round(w0 * ratio[1])), int(round(h0 * ratio[0]))
            ir = cv2.resize(ir, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((H, W), 114, dtype=np.uint8)
            top, left = int(round(padh - 0.1)), int(round(padw - 0.1))  # same rounding as Ultralytics LetterBox
            canvas[top:top + nh, left:left + nw] = ir
            t = torch.from_numpy(canvas).to(rgb.device).float().unsqueeze(0) / 255.0
            irs.append(t.repeat(3, 1, 1))
        batch["img"] = RGBTBatch((rgb, torch.stack(irs, 0)))
        return batch


def run(variant, device):
    pre = YOLO(WEIGHTS).model

    class T(RGBTDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pre

        if variant in ("A", "B"):
            def build_dataset(self, img_path, mode="train", batch=None):  # stock Ultralytics: val pad=0.5
                return DetectionTrainer.build_dataset(self, img_path, mode, batch)

        if variant == "B":
            def get_validator(self):
                self.loss_names = "box_loss", "cls_loss", "dfl_loss"
                from copy import copy
                return LetterboxIRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                            _callbacks=self.callbacks, trainer=self)

    tr = T(overrides=dict(model="yolo11n.pt", data=rsr.DATA_YAML, epochs=1, imgsz=960, batch=4, device=device,
                          workers=0, conf=0.001, iou=0.7, plots=False, project=os.path.join(OUT, "_scratch_runs"),
                          name=f"crosscheck_{variant}", exist_ok=True, val=True, rect=True, box=10.0, cls=0.3, dfl=1.5))
    tr._setup_train()
    tr.loss_items = torch.zeros(3, device=device)
    tr.epoch = 0
    tr.loss = torch.zeros(1, device=device)
    shape = tuple(next(iter(tr.test_loader))["img"].shape)
    tr.validate()
    box = tr.validator.metrics.box
    idx = int(np.argmin(np.abs(box.px - rsr.FIXED_CONF)))
    res = dict(mAP50=float(box.map50), mAP50_95=float(box.map), P=float(box.p_curve[0][idx]), R=float(box.r_curve[0][idx]))
    del tr, pre
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return shape, res


if __name__ == "__main__":
    dev = rsr.get_device()
    rows = {}
    for v in ("A", "B", "C"):
        rows[v] = run(v, dev)
        print(f"\n### {v}: val tensor {rows[v][0]} -> {rows[v][1]}\n", flush=True)
    print("\n==== CROSS-CHECK SUMMARY ====", WEIGHTS)
    for v, label in (("A", "pad0.5, IR stretched (old)"), ("B", "pad0.5, IR letterboxed"), ("C", "pad0.0 (fix)")):
        s, r = rows[v]
        print(f"{v} {label:28s} tensor={s} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f} P={r['P']:.3f} R={r['R']:.3f}")
