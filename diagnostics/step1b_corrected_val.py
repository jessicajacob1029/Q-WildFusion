"""
Step 1 (corrected, v2): the standalone validator(model=...) call fails because
BaseValidator wraps a bare model in AutoBackend, which cannot accept a tuple
input at all (see the AttributeError this script's first version hit). The
actual code path that DOES call the raw model directly (no AutoBackend) is
BaseTrainer.validate() -> self.validator(self) i.e. the `trainer=` branch.

So: reuse RGBTDetectionTrainer's real setup machinery (dataloader, args,
save_dir, etc.) so trainer.validate() is invoked exactly like it is during
real training, but (a) load the model from best.pt UNCHANGED instead of
rebuilding+reinitializing it via get_model(weights=...) (which would discard
the trained fusion_backbone weights and re-seed them from the base yolo11n.pt
via initialize_fusion_backbone_from_yolo -- not what we want to evaluate),
and (b) swap in RGBTDiagnosticValidator (defined here only) whose preprocess()
fixes the tuple-ification bug found in step1a.

No file under phase3/code/ is modified.
"""
import os
import sys
import time
from copy import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import RANK

from phase3.code.rgbt_trainer import RGBTDetectionTrainer, RGBTBatch

print("ultralytics version:", ultralytics.__version__)
print("torch version:", torch.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

WEIGHTS = "runs/detect/abl_no_cbam_add/weights/best.pt"
DATA = "phase3/dataset_aligned/data.yaml"

PRELOADED_MODEL = YOLO(WEIGHTS).model
print("loaded checkpoint model type:", type(PRELOADED_MODEL).__name__)
print("fusion_backbone present:", hasattr(PRELOADED_MODEL, "fusion_backbone"))


class RGBTDiagnosticValidator(DetectionValidator):
    """DetectionValidator with the one fix: preprocess() also loads the
    matching IR image and hands the model the (rgb, ir) tuple it expects."""

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        rgb = batch["img"]
        ir_root = self.data.get("ir")
        if not os.path.isabs(str(ir_root)):
            ir_root = os.path.abspath(os.path.join(self.data.get("path"), str(ir_root)))

        ir_images = []
        for rgb_path in batch["im_file"]:
            ir_path = RGBTDetectionTrainer._matching_ir_path(str(rgb_path), ir_root)
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                raise FileNotFoundError(f"Missing thermal image: {ir_path}")
            ir = cv2.resize(ir, (rgb.shape[-1], rgb.shape[-2]), interpolation=cv2.INTER_LINEAR)
            ir = torch.from_numpy(ir).to(rgb.device)
            ir = ir.float().unsqueeze(0) / 255.0
            ir_images.append(ir.repeat(3, 1, 1))

        batch["img"] = RGBTBatch((rgb, torch.stack(ir_images, dim=0)))
        return batch


class DiagnosticTrainer(RGBTDetectionTrainer):
    fusion_mode = "add"
    use_cbam = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Ignore cfg/weights entirely -- return the exact, already fully
        # trained checkpoint object, unmodified.
        return PRELOADED_MODEL

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return RGBTDiagnosticValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


t0 = time.time()
trainer = DiagnosticTrainer(overrides={
    "model": "yolo11n.pt",  # placeholder; get_model() ignores this and returns PRELOADED_MODEL
    "data": DATA,
    "epochs": 1,
    "imgsz": 960,
    "batch": 4,
    "device": device,
    "workers": 0,
    "conf": 0.001,
    "iou": 0.7,
    "plots": False,
    "project": "diagnostics/_scratch_runs",
    "name": "step1b_corrected_val",
    "exist_ok": True,
    "val": True,
    "rect": True,
    "box": 10.0, "cls": 0.3, "dfl": 1.5,
})
trainer._setup_train()
trainer.loss_items = torch.zeros(3, device=device)  # normally set inside the train() loop, which we skip
trainer.epoch = 0  # ditto
trainer.loss = torch.zeros(1, device=device)  # ditto (only used as a fallback default, never actually read)
t_setup = time.time()
print(f"setup wall-clock: {t_setup - t0:.1f}s")

metrics, fitness = trainer.validate()
t1 = time.time()

print()
print(f"wall-clock for corrected validation pass: {t1 - t_setup:.1f}s")
print()
print("=== corrected validation metrics (conf=0.001, fusion_backbone ACTUALLY engaged) ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

try:
    import numpy as np
    box = trainer.validator.metrics.box
    print()
    print("precision at best-F1 threshold (metrics/precision(B)):", float(box.p[0]))
    print("recall at best-F1 threshold (metrics/recall(B)):", float(box.r[0]))
    px = box.px  # confidence axis, 1000 points from 0 to 1
    r_curve = box.r_curve[0]  # recall as a function of confidence, class 0 ("human")
    p_curve = box.p_curve[0]
    print("recall ceiling (max recall over ALL confidence thresholds, i.e. at conf->0):", float(r_curve.max()))
    idx_025 = int(np.argmin(np.abs(px - 0.25)))
    print(f"recall at conf={px[idx_025]:.3f} (~0.25):", float(r_curve[idx_025]))
    print(f"precision at conf={px[idx_025]:.3f} (~0.25):", float(p_curve[idx_025]))
except Exception as e:
    import traceback
    print("could not extract explicit recall-curve arrays:")
    traceback.print_exc()
