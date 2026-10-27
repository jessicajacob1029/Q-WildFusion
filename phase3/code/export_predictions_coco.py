import os
import json
import argparse
from ultralytics import YOLO
from PIL import Image

def export_coco_predictions(weights, images_dir, output_json, conf=0.001):
    model = YOLO(weights)

    coco_preds = []
    image_id = 1
    ann_id = 1

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg")
    ])

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)

        with Image.open(img_path) as img:
            width, height = img.size

        results = model(img_path, conf=conf, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            image_id += 1
            continue

        boxes = results[0].boxes

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            score = float(b.conf[0])

            coco_preds.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2)
                ],
                "score": round(score, 4)
            })
            ann_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_preds, f, indent=2)

    print(f"✅ Saved COCO predictions to {output_json}")
    print(f"Images processed: {len(image_files)}")
    print(f"Total predictions: {len(coco_preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    export_coco_predictions(
        args.weights,
        args.images,
        args.output,
        conf=args.conf
    )
