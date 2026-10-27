import os
import sys

# Increase MPS memory headroom on Apple Silicon to avoid OOM during large model training
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO
from phase3.code.rgbt_trainer import RGBTDetectionTrainer

def main():
    print("[INFO] Starting MAXIMAL CONFIG CBAM + VQC Training (Target: 90%+ mAP)")

    # Use the absolute largest YOLOv11 model (Extra Large) for maximum capacity
    yolo = YOLO("yolo11x.pt")

    class Proposed90PlusTrainer(RGBTDetectionTrainer):
        fusion_mode = "vqc"
        use_cbam = True

    # Train with extreme accuracy settings (will be slower but yield max AP)
    yolo.train(
        trainer=Proposed90PlusTrainer,
        data="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml",

        epochs=10,           # Reduced for quick results
        imgsz=640,           # Smaller resolution for faster training
        batch=2,             # Increased batch for speed

        # Aggressive loss weighting
        box=7.5,
        cls=1.5,
        dfl=2.5,

        # We keep spatial align but use heavy color/lighting aug 
        # (YOLO automatically drops mosaic if close_mosaic is hit)
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
        mosaic=0.5,          # Light mosaic to prevent overfitting
        mixup=0.1,
        copy_paste=0.0,

        rect=False,
        close_mosaic=5,     # Turn off mosaic for last 5 epochs

        patience=50,
        device="mps",        # Assuming running on Mac Silicon
        workers=4,

        name="yolov11X_vqc_cbam_quick",

        # Slower, deeper learning rate schedule
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
