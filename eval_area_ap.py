import json
import os
from PIL import Image

pred_path = "runs/detect/val12/predictions.json"
img_dir = "phase3/dataset_aligned/images/rgb/val"
lbl_dir = "phase3/dataset_aligned/labels/rgb/val"

preds = json.load(open(pred_path))

# area thresholds for imgsz=960
SMALL = 48*48
LARGE = 144*144

tp_large = 0
gt_large = 0

tp_medium = 0
gt_medium = 0

files = sorted(os.listdir(img_dir))

# count GT objects
for f in files:
    if not f.endswith(".jpg"):
        continue

    img = Image.open(os.path.join(img_dir,f))
    w,h = img.size

    label_path = os.path.join(lbl_dir,f.replace(".jpg",".txt"))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as lf:
        for line in lf:
            cls,x,y,bw,bh = map(float,line.split())
            area = bw*bh*w*h

            if area >= LARGE:
                gt_large += 1
            elif area >= SMALL:
                gt_medium += 1

# count predictions
for p in preds:
    area = p["bbox"][2]*p["bbox"][3]

    if area >= LARGE:
        tp_large += 1
    elif area >= SMALL:
        tp_medium += 1

print("\nMedium objects (approx AP50 proxy):", tp_medium/gt_medium if gt_medium else 0)
print("Large objects (approx AP50 proxy):", tp_large/gt_large if gt_large else 0)
