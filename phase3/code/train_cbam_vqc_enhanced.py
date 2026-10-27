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
    print("[INFO] Starting ENHANCED CBAM + VQC YOLOv11 Training (Proposed Model - Target: 90%+ mAP)")

    # Load YOLOv11 base with pretrained weights
    yolo = YOLO(os.path.join(ROOT, "yolo11x.pt"))  # Use local pretrained weights for better starting point

    class ProposedRGBTTrainer(RGBTDetectionTrainer):
        fusion_mode = "vqc"
        use_cbam = True

    # Train with highly optimized settings for 90%+ mAP
    yolo.train(
        trainer=ProposedRGBTTrainer,
        data=os.path.join(ROOT, "phase3", "dataset_aligned", "data.yaml"),

        epochs=10,  # Reduced for quick results
        imgsz=640,
        batch=2,  # Smaller batch for stability

        # Fine-tuned loss weights (optimized for high accuracy)
        box=7.5,    # Lower box loss weight
        cls=1.5,    # Higher classification weight
        dfl=2.5,    # Higher DFL weight for better localization

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

        patience=50,
        device="mps",
        workers=4,

        name="cbam_vqc_enhanced_quick",

        # Advanced learning rate schedule
        lr0=0.0001,  # Lower initial LR for stability
        lrf=0.001,   # Lower final LR
        momentum=0.98,  # Higher momentum
        weight_decay=0.0005,  # Lower weight decay
        warmup_epochs=2.0,  # Shorter warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,

        # Additional optimizations
        cos_lr=True,  # Cosine annealing
        label_smoothing=0.1,  # Label smoothing for better generalization
    )

if __name__ == "__main__":
    main()
