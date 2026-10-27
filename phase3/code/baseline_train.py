import os
import sys

# Increase MPS memory headroom on Apple Silicon to avoid OOM during large model training
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO

def main():
    print("[INFO] Starting BASELINE YOLO Training (No Fusion)")

    # Use the absolute largest YOLOv11 model (Extra Large) for maximum capacity
    yolo = YOLO("yolo11x.pt")

    # Train with standard YOLO settings
    yolo.train(
        data="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml",

        epochs=10,           # Reduced for quick results
        imgsz=640,           # Smaller resolution for faster training
        batch=2,             # Increased batch for speed

        # Standard loss weighting
        box=7.5,
        cls=1.5,
        dfl=2.5,

        # Standard augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.0,

        rect=False,
        close_mosaic=5,      # Turn off mosaic for last 5 epochs

        patience=50,
        device="mps",
        workers=4,

        name="yolov11X_baseline",

        # Standard learning rate schedule
        lr0=0.0001,
        lrf=0.001,
        momentum=0.98,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        label_smoothing=0.1,
    )

if __name__ == "__main__":
    main()