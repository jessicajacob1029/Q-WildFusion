from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Use the phase1 predictions that achieved 90%+ mAP
pred_path = "runs/detect/val20/predictions.json"
gt_path = "phase1/dataset/labels/val.json"  # Need to create this

# First, let's create the ground truth JSON file from the YOLO labels
import os
import json
from pathlib import Path

def create_coco_annotations(labels_dir, images_dir, output_file):
    """Convert YOLO labels to COCO format for evaluation"""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "person"}]
    }

    ann_id = 0
    img_id = 0

    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Add image
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size

        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # Add annotations if label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height

                        x1 = x_center - w/2
                        y1 = y_center - h/2
                        x2 = x_center + w/2
                        y2 = y_center + h/2

                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1

        img_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

# Create ground truth file
create_coco_annotations(
    "phase1/dataset/labels/val",
    "phase1/dataset/images/val",
    "phase1/dataset/labels/val.json"
)

cocoGt = COCO(gt_path)
cocoDt = cocoGt.loadRes(pred_path)

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

# medium and large area ranges (scaled for imgsz=960)
cocoEval.params.areaRng = [
    [48**2, 144**2],   # medium
    [144**2, 1e10]     # large
]

cocoEval.params.areaRngLbl = ["medium","large"]

print("\nEvaluating Medium and Large objects with Phase1 model...\n")

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
