import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------
MODEL_PATH = "runs/detect/yolov11L_rgb_ir_cbam_stage22/weights/best.pt"
IMAGE_PATH = "phase3/dataset_aligned/images/rgb/val/150096.jpg"

yolo = YOLO(MODEL_PATH)
model = yolo.model
model.eval()

# --------------------------------------------------
# Select target layer (SAFE backbone layer)
# --------------------------------------------------
# Backbone / neck layer → good semantic gradients
target_layers = [model.model[10]]

# --------------------------------------------------
# Load and preprocess image
# --------------------------------------------------
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found!"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0

input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
input_tensor.requires_grad_(True)

# --------------------------------------------------
# Custom YOLO target for Grad-CAM
# --------------------------------------------------
class YOLOConfidenceTarget:
    def __call__(self, model_output):
        """
        model_output shape: [B, N, C]
        Use max object confidence across all detections
        """
        if model_output.ndim == 3:
            scores = model_output[..., 4]  # objectness score
            return scores.max()
        return model_output.sum()

targets = [YOLOConfidenceTarget()]

# --------------------------------------------------
# Run Grad-CAM
# --------------------------------------------------
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

with torch.enable_grad():
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

# --------------------------------------------------
# Overlay heatmap
# --------------------------------------------------
cam_image = show_cam_on_image(
    img_norm,
    grayscale_cam,
    use_rgb=True
)

output_path = "gradcam_output.jpg"
cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

print(f"✅ Grad-CAM saved to {output_path}")
