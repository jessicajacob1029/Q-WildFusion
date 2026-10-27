"""
Diagnostic (read-only w.r.t. the codebase): does validation, as invoked by
Ultralytics' own BaseTrainer.validate() -> self.validator(self), actually feed
RGBTDetectionModel.forward() a (rgb, ir) tuple (triggering predict_rgbt / the
fusion_backbone), or a plain RGB tensor (silently falling through to the stock
DetectionModel.forward on self.model[0:23] alone, bypassing fusion_backbone
entirely)?

Method: monkeypatch RGBTDetectionModel.forward in-memory (this process only,
nothing written to disk, no file under phase3/code is modified) to record the
type of `x` on every call, tagged with whether the model was in eval mode
(model.training == False => this is a validation-time forward call) or train
mode. Then run the smallest possible real training call (1 epoch, tiny
fraction, small imgsz) so a real validation pass actually happens, using the
exact same RGBTDetectionTrainer class the ablation grid used.
"""
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ultralytics import YOLO
import ultralytics

from phase3.code.rgbt_trainer import RGBTDetectionTrainer
from phase3.code.model.yolov11_rgbt import RGBTDetectionModel

print("ultralytics version:", ultralytics.__version__)
print("torch version:", torch.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device selected for this diagnostic run:", device)

calls = []
_orig_forward = RGBTDetectionModel.forward


def _patched_forward(self, x, *args, **kwargs):
    calls.append({
        "eval_mode": not self.training,
        "x_type": type(x).__name__,
        "is_tuple": isinstance(x, tuple),
        "is_dict": isinstance(x, dict),
    })
    return _orig_forward(self, x, *args, **kwargs)


RGBTDetectionModel.forward = _patched_forward

t0 = time.time()
yolo = YOLO("yolo11n.pt")
ProbeTrainer = type("ProbeTrainer", (RGBTDetectionTrainer,), {"fusion_mode": "add", "use_cbam": False})
yolo.train(
    trainer=ProbeTrainer,
    data="phase3/dataset_aligned/data.yaml",
    epochs=1,
    imgsz=320,
    batch=2,
    fraction=0.01,
    device=device,
    workers=0,
    project="diagnostics/_scratch_runs",
    name="step1a_val_input_type",
    exist_ok=True,
    patience=100,
    plots=False,
    val=True,
    rect=True,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
    scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0,
)
t1 = time.time()

RGBTDetectionModel.forward = _orig_forward  # restore, though process exits anyway

print()
print(f"wall-clock for this diagnostic run: {t1 - t0:.1f}s")
print(f"total forward() calls recorded: {len(calls)}")

train_calls = [c for c in calls if not c["eval_mode"]]
eval_calls = [c for c in calls if c["eval_mode"]]
print(f"train-mode forward calls: {len(train_calls)}")
print(f"eval-mode (validation) forward calls: {len(eval_calls)}")

if train_calls:
    from collections import Counter
    print("train-mode x_type distribution:", Counter(c["x_type"] for c in train_calls))
    print("train-mode is_tuple distribution:", Counter(c["is_tuple"] for c in train_calls))

if eval_calls:
    from collections import Counter
    print("EVAL-mode x_type distribution:", Counter(c["x_type"] for c in eval_calls))
    print("EVAL-mode is_tuple distribution:", Counter(c["is_tuple"] for c in eval_calls))
    print()
    if all(c["is_tuple"] for c in eval_calls):
        print("VERDICT: validation DOES pass a tuple -> fusion_backbone IS exercised during val.")
    elif not any(c["is_tuple"] for c in eval_calls):
        print("VERDICT: validation NEVER passes a tuple -> fusion_backbone is BYPASSED during val "
              "(falls through to stock DetectionModel.forward on self.model[0:23] alone).")
    else:
        print("VERDICT: MIXED — some eval calls got a tuple, some didn't. Inspect calls list.")
else:
    print("VERDICT: INCONCLUSIVE — no eval-mode forward() calls were recorded at all.")
