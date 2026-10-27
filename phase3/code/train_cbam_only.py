import os
import sys

# -----------------------------
# Path setup
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO
from phase3.code.rgbt_trainer import RGBTDetectionTrainer

# -----------------------------
# Main training
# -----------------------------
def main():
    print("[INFO] Starting CBAM-Only YOLOv11 Training (CBAM + Add Fusion, No VQC)")

    # Load YOLOv11 base
    yolo = YOLO(os.path.join(ROOT, "yolo11x.pt"))

    class CBAMOnlyRGBTTrainer(RGBTDetectionTrainer):
        fusion_mode = "add"
        use_cbam = True

    # Train with optimized settings
    yolo.train(
        trainer=CBAMOnlyRGBTTrainer,
        data=os.path.join(ROOT, "phase3", "dataset_aligned", "data.yaml"),

        epochs=10,
        imgsz=640,
        batch=2,

        # Optimized loss weights
        box=7.5,
        cls=1.5,
        dfl=2.5,

        # Keep paired RGB/thermal images spatially aligned.
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        rect=True,
        close_mosaic=5,

        patience=30,
        device="mps",
        workers=4,

        name="cbam_only_quick",

        # Learning rate schedule
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

if __name__ == "__main__":
    main()
