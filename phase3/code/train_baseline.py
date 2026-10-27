import os
import sys
import torch
import torch.nn as nn
import argparse

# -----------------------------
# Path setup
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO

# -----------------------------
# Baseline Model (No CBAM, No VQC)
# -----------------------------
class YOLOv11_Baseline(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()

        # Use standard YOLOv11 backbone (RGB only)
        self.backbone = yolo_model.model.model[:11]  # Up to backbone output
        self.neck = yolo_model.model.model[11:23]
        self.head = yolo_model.model.model[23]

        # Required by Ultralytics
        self.yaml = yolo_model.model.yaml
        self.names = yolo_model.model.names
        self.nc = yolo_model.model.yaml["nc"]

    def forward(self, x):
        """
        x: RGB image tensor [B, 3, H, W]
        """
        # Standard YOLO forward pass
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)

# -----------------------------
# Main training
# -----------------------------
def main():
    print("[INFO] Starting BASELINE YOLOv11 Training (No CBAM, No VQC)")

    # Load YOLOv11 base
    yolo = YOLO("yolo11s.yaml")

    # Replace model with baseline wrapper
    yolo.model = YOLOv11_Baseline(yolo).to(yolo.device)

    # Train with optimized settings
    yolo.train(
        data="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml",  # Use absolute path

        epochs=50,
        imgsz=960,
        batch=8,

        # Optimized loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Data augmentation
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        rect=True,
        close_mosaic=10,

        patience=20,
        device="mps",
        workers=4,

        name="baseline_no_cbam",

        # Learning rate schedule
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

if __name__ == "__main__":
    main()