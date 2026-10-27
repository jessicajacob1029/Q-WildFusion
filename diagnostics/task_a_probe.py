"""
Task A - decisive runtime probe: does the REAL RGBTDetectionTrainer's
get_validator() actually install RGBTDetectionValidator (the corrected,
tuple-ifying preprocess) as the trainer-time / checkpoint-selection /
patience validator, or does end-of-epoch validation during real training
still go through stock DetectionValidator.preprocess (bare RGB)?

No source files edited. All instrumentation is runtime monkeypatch, inside
this throwaway script only. Runs ONE real 1-epoch training call (tiny
fraction) through the unmodified RGBTDetectionTrainer and logs, at the
end-of-epoch validation pass:
  - which preprocess() implementation actually executes (tag by class)
  - what structure reaches RGBTDetectionModel.forward() during that
    validation pass (single tensor vs 2-tuple, shapes)
"""
import inspect
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from phase3.code.rgbt_trainer import RGBTDetectionTrainer, RGBTDetectionValidator
from phase3.code.model.yolov11_rgbt import RGBTDetectionModel

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------------------------------------------------------
# 1. Static structure
# ---------------------------------------------------------------------------
print("=" * 70)
print("STATIC: RGBTDetectionValidator.preprocess source")
print("=" * 70)
print(inspect.getsource(RGBTDetectionValidator.preprocess))

print("=" * 70)
print("STATIC: RGBTDetectionTrainer.get_validator source")
print("=" * 70)
print(inspect.getsource(RGBTDetectionTrainer.get_validator))

# ---------------------------------------------------------------------------
# 2. Runtime probe
# ---------------------------------------------------------------------------
calls = []

_orig_corrected_preprocess = RGBTDetectionValidator.preprocess
_orig_stock_preprocess = DetectionValidator.preprocess
_orig_forward = RGBTDetectionModel.forward


def _patched_corrected_preprocess(self, batch):
    out = _orig_corrected_preprocess(self, batch)
    img = out["img"]
    calls.append({
        "tag": "CORRECTED(RGBTDetectionValidator).preprocess",
        "self_type": type(self).__name__,
        "img_type": type(img).__name__,
        "img_len": len(img) if isinstance(img, tuple) else None,
        "shapes": [tuple(t.shape) for t in img] if isinstance(img, tuple) else tuple(img.shape),
    })
    return out


def _patched_stock_preprocess(self, batch):
    out = _orig_stock_preprocess(self, batch)
    img = out["img"]
    calls.append({
        "tag": "STOCK(DetectionValidator).preprocess",
        "self_type": type(self).__name__,
        "img_type": type(img).__name__,
        "img_len": len(img) if isinstance(img, tuple) else None,
        "shapes": [tuple(t.shape) for t in img] if isinstance(img, tuple) else tuple(img.shape),
    })
    return out


forward_calls = []


def _patched_forward(self, x, *args, **kwargs):
    if not self.training:  # eval mode == validation-time forward
        forward_calls.append({
            "x_type": type(x).__name__,
            "is_tuple": isinstance(x, tuple),
            "shapes": [tuple(t.shape) for t in x] if isinstance(x, tuple) else (tuple(x.shape) if hasattr(x, "shape") else None),
        })
    return _orig_forward(self, x, *args, **kwargs)


RGBTDetectionValidator.preprocess = _patched_corrected_preprocess
DetectionValidator.preprocess = _patched_stock_preprocess
RGBTDetectionModel.forward = _patched_forward

print("=" * 70)
print("RUNTIME PROBE: real 1-epoch training call through unmodified RGBTDetectionTrainer")
print("=" * 70)

yolo = YOLO("yolo11n.pt")
try:
    yolo.train(
        trainer=RGBTDetectionTrainer,
        data="phase3/dataset_aligned/data.yaml",
        epochs=1,
        imgsz=320,
        batch=2,
        fraction=0.02,
        device=device,
        workers=0,
        project="diagnostics/tmp_probe",
        name="run",
        exist_ok=True,
        patience=100,
        plots=False,
        val=True,
        rect=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0,
    )
finally:
    RGBTDetectionValidator.preprocess = _orig_corrected_preprocess
    DetectionValidator.preprocess = _orig_stock_preprocess
    RGBTDetectionModel.forward = _orig_forward

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"preprocess() calls captured: {len(calls)}")
from collections import Counter
tag_counts = Counter(c["tag"] for c in calls)
for tag, n in tag_counts.items():
    print(f"  {tag}: {n} calls")
if calls:
    print("  first call detail:", calls[0])
    print("  last call detail:", calls[-1])

print()
print(f"eval-mode (validation-time) RGBTDetectionModel.forward() calls captured: {len(forward_calls)}")
if forward_calls:
    x_type_counts = Counter(c["x_type"] for c in forward_calls)
    is_tuple_counts = Counter(c["is_tuple"] for c in forward_calls)
    print("  x_type distribution:", dict(x_type_counts))
    print("  is_tuple distribution:", dict(is_tuple_counts))
    print("  sample shapes (first call):", forward_calls[0]["shapes"])
else:
    print("  NONE CAPTURED -- inconclusive, no eval-mode forward() calls observed")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
corrected_fired = tag_counts.get("CORRECTED(RGBTDetectionValidator).preprocess", 0) > 0
all_eval_forward_tuple = bool(forward_calls) and all(c["is_tuple"] for c in forward_calls)
if corrected_fired and all_eval_forward_tuple:
    print("PASS: corrected preprocess fired during trainer-time validation AND "
          "forward() received a paired (rgb, ir) tuple every time.")
elif not calls:
    print("INCONCLUSIVE: no preprocess() calls captured at all.")
else:
    print("FAIL: either corrected preprocess never fired as the trainer-time validator, "
          "or forward() received something other than a paired tuple during validation. "
          "See tag_counts / forward_calls above.")
