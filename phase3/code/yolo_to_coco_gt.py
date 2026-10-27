import os
import json
import argparse
import re
from PIL import Image

def yolo_to_coco(images_dir, labels_dir, output_json):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "object"}]

    ann_id = 1
    img_id = 1

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

        with Image.open(img_path) as img:
            width, height = img.size

        match = re.search(r'\d+', img_name)
        if match:
            current_img_id = int(match.group(0))
        else:
            current_img_id = img_id

        images.append({
            "id": current_img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0]) + 1
                        xc, yc, w, h = map(float, parts[1:])

                        bbox_w = w * width
                        bbox_h = h * height
                        x_min = (xc * width) - bbox_w / 2
                        y_min = (yc * height) - bbox_h / 2

                        annotations.append({
                            "id": ann_id,
                            "image_id": current_img_id,
                            "category_id": class_id,
                            "bbox": [
                                round(x_min, 2),
                                round(y_min, 2),
                                round(bbox_w, 2),
                                round(bbox_h, 2)
                            ],
                            "area": round(bbox_w * bbox_h, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ COCO ground truth saved to {output_json}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    yolo_to_coco(args.images, args.labels, args.output)
