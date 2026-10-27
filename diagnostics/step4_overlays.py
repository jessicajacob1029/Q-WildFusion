"""
Step 4: qualitative overlays. GT boxes in green, baseline best.pt predictions
(conf=0.001, fusion_backbone correctly engaged via the same RGBTBatch fix
used in step1b) in red. Saved to diagnostics/overlays/, not judged here.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import nms

from phase3.code.rgbt_trainer import RGBTDetectionTrainer

WEIGHTS = "runs/detect/abl_no_cbam_add/weights/best.pt"
IMG_DIR = "phase3/dataset_aligned/images/rgb/val"
LBL_DIR = "phase3/dataset_aligned/labels/rgb/val"
IR_ROOT = os.path.abspath("phase3/dataset_aligned/images/ir")
OUT_DIR = "diagnostics/overlays"
IMGSZ = 960
CONF = 0.001
IOU = 0.7

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = YOLO(WEIGHTS).model
model.to(device)
model.eval()
print("model type:", type(model).__name__, "fusion_backbone present:", hasattr(model, "fusion_backbone"))

val_images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
picks = [val_images[0], val_images[len(val_images) // 2], val_images[-1]]
print("picked:", picks)


def load_and_preprocess(fname):
    rgb_path = os.path.join(IMG_DIR, fname)
    rgb_orig = cv2.imread(rgb_path)  # BGR, original size
    h0, w0 = rgb_orig.shape[:2]
    rgb_resized = cv2.resize(rgb_orig, (IMGSZ, IMGSZ))
    rgb_t = torch.from_numpy(rgb_resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    rgb_t = rgb_t.unsqueeze(0).to(device)

    ir_path = RGBTDetectionTrainer._matching_ir_path(os.path.abspath(rgb_path), IR_ROOT)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.resize(ir, (IMGSZ, IMGSZ))
    ir_t = torch.from_numpy(ir).float().unsqueeze(0) / 255.0
    ir_t = ir_t.repeat(3, 1, 1).unsqueeze(0).to(device)

    return rgb_orig, (h0, w0), rgb_t, ir_t


def load_gt_boxes_xyxy(fname, h0, w0):
    stem = os.path.splitext(fname)[0]
    lbl_path = os.path.join(LBL_DIR, stem + ".txt")
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, x, y, w, h = map(float, line.split())
            cx, cy, bw, bh = x * w0, y * h0, w * w0, h * h0
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            boxes.append((x1, y1, x2, y2))
    return boxes


os.makedirs(OUT_DIR, exist_ok=True)

for fname in picks:
    rgb_orig, (h0, w0), rgb_t, ir_t = load_and_preprocess(fname)

    with torch.no_grad():
        preds = model.predict_rgbt((rgb_t, ir_t))
        preds_nms = nms.non_max_suppression(preds, conf_thres=CONF, iou_thres=IOU, max_det=300)[0]
    # preds_nms: [N, 6] = x1,y1,x2,y2,conf,cls in IMGSZ-scale coords -> rescale to original image size
    scale_x, scale_y = w0 / IMGSZ, h0 / IMGSZ

    canvas = rgb_orig.copy()

    gt_boxes = load_gt_boxes_xyxy(fname, h0, w0)
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    n_preds = 0 if preds_nms is None else preds_nms.shape[0]
    if preds_nms is not None:
        for det in preds_nms.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.putText(canvas, f"{conf:.2f}", (int(x1), max(0, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    out_path = os.path.join(OUT_DIR, f"overlay_{fname}")
    cv2.imwrite(out_path, canvas)
    print(f"{fname}: GT boxes={len(gt_boxes)} predictions(conf>=0.001)={n_preds} -> saved {out_path}")

print("done")
