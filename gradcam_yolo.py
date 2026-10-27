import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load YOLO
yolo = YOLO("runs/detect/yolov11L_rgb_ir_cbam_highAP/weights/best.pt")
model = yolo.model
model.eval()

# enable gradients
for p in model.parameters():
    p.requires_grad = True

# pick a deep conv layer
target_layers = [model.model[-2]]

# image
image_path = "phase3/dataset_aligned/images/rgb/val/150096.jpg"

img = cv2.imread(image_path)
img = cv2.resize(img,(640,640))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_img = np.float32(rgb_img)/255.0

input_tensor = torch.from_numpy(rgb_img.transpose(2,0,1)).unsqueeze(0)

class YOLOScoreTarget:
    def __call__(self, model_output):
        # use highest detection score as GradCAM target
        scores = model_output[...,4]
        return scores.max()

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        preds = self.model(x)
        return preds[0]

wrapped_model = YOLOWrapper(model)

targets = [YOLOScoreTarget()]

cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

cv2.imwrite("gradcam_output.jpg", visualization)

print("GradCAM saved: gradcam_output.jpg")
