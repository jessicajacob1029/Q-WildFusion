from ultralytics import YOLO
import os
import numpy as np

MODEL_PATH = "runs/detect/rgb_ir_cbam_final2/weights/best.pt"
DATA_YAML = "phase3/dataset_aligned/data.yaml"
AREA_THRESH = 0.02  # medium + large only

model = YOLO(MODEL_PATH)

# 1. Run prediction on validation split
print("Running YOLOv11 prediction on validation set...")
preds = model.predict(
    data=DATA_YAML,
    split="val",
    save=False,
    conf=0.001,
    iou=0.6
)

print("Filtering boxes by area...")

kept = 0
total = 0

for r in preds:
    h, w = r.orig_shape

    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()

    for x1, y1, x2, y2 in boxes:
        total += 1
        area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
        if area_ratio >= AREA_THRESH:
            kept += 1

print("\n===== FILTER SUMMARY =====")
print(f"Boxes kept (medium+large): {kept}/{total}")
print("\nNow compute mAP@50 using filtered boxes (see explanation below).")
