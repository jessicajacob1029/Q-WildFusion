import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

MODEL_PATH = "runs/detect/yolov11L_rgb_ir_cbam_highAP/weights/best.pt"
IMAGE_DIR = "phase3/dataset_aligned/images/rgb/val"
OUTPUT_DIR = "runs/detect/qualitative_results_gradcam"

# Load model
print(f"Loading model from {MODEL_PATH}")
yolo = YOLO(MODEL_PATH)
model = yolo.model
model.eval()

target_layers = [model.model[10]]

class YOLOConfidenceTarget:
    def __call__(self, model_output):
        scores = None
        if isinstance(model_output, tuple):
            out = model_output[1]
            if isinstance(out, dict):
                scores = out.get("scores")
        if scores is None and torch.is_tensor(model_output):
            if model_output.ndim == 3:
                scores = model_output[..., 4]
        if scores is None:
            raise ValueError("Could not extract YOLO scores from model output")
        return scores.max()

cam = GradCAM(model=model, target_layers=target_layers)

os.makedirs(OUTPUT_DIR, exist_ok=True)
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
if len(image_paths) == 0:
    raise ValueError(f"No JPG images found in {IMAGE_DIR}")

print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

for idx, image_path in enumerate(image_paths, start=1):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Skipping missing image: {image_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    input_tensor.requires_grad_(True)

    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=[YOLOConfidenceTarget()])[0]

    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    cam_image = (cam_image * 255).astype(np.uint8) if cam_image.dtype == np.float32 else cam_image

    # Add fixed human label text
    cv2.putText(
        cam_image,
        "human",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_path = Path(OUTPUT_DIR) / image_path.name
    cv2.imwrite(str(output_path), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    if idx % 100 == 0:
        print(f"Saved {idx}/{len(image_paths)} GradCAM images")

print(f"✅ Saved {len(image_paths)} GradCAM heatmap images to {OUTPUT_DIR}")
