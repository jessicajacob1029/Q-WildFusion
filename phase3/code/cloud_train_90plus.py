import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO
from phase3.code.rgbt_trainer import RGBTDetectionTrainer


def main():
    print("[INFO] Starting CLOUD CBAM + VQC Training (Target: 90%+ mAP)")

    # Use the largest YOLOv11 model for maximum capacity
    yolo = YOLO(os.path.join(ROOT, "yolo11x.pt"))

    class Proposed90PlusTrainer(RGBTDetectionTrainer):
        fusion_mode = "vqc"
        use_cbam = True

    yolo.train(
        trainer=Proposed90PlusTrainer,
        data=os.path.join(ROOT, "phase3", "dataset_aligned", "data.yaml"),

        epochs=150,
        imgsz=960,
        batch=4,

        # Aggressive loss weighting
        box=7.5,
        cls=1.5,
        dfl=2.5,

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
        close_mosaic=20,

        patience=50,
        device="cuda",
        workers=8,

        name="yolov11x_vqc_cbam_cloud",

        lr0=0.0001,
        lrf=0.001,
        momentum=0.98,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        cos_lr=True,
        label_smoothing=0.1,
    )


if __name__ == "__main__":
    main()
