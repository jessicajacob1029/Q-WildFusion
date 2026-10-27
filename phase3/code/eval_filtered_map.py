

from ultralytics import YOLO
import numpy as np

MODEL_PATH = "runs/detect/rgb_ir_cbam_final2/weights/best.pt"
DATA_YAML = "phase3/dataset_aligned/data.yaml"
AREA_THRESH = 0.02  # 2% of image


model = YOLO(MODEL_PATH)

# Run normal validation (we'll intercept results)
results = model.val(data=DATA_YAML, save=False, plots=False)

# Pseudocode: iterate predictions & GT
filtered_preds = []
filtered_gts = []

for img_result in results:
	img_h, img_w = img_result.orig_shape

	for box in img_result.boxes:
		x1, y1, x2, y2 = box.xyxy[0]
		area = (x2 - x1) * (y2 - y1)
		if area / (img_h * img_w) >= AREA_THRESH:
			filtered_preds.append(box)

	for gt in img_result.gt_boxes:
		gx1, gy1, gx2, gy2 = gt
		gt_area = (gx2 - gx1) * (gy2 - gy1)
		if gt_area / (img_h * img_w) >= AREA_THRESH:
			filtered_gts.append(gt)

# Now compute mAP on filtered_preds vs filtered_gts
# (we will plug in Ultralytics metrics utilities next)
