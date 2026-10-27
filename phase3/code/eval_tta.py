import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO
from phase3.code.rgbt_trainer import RGBTDetectionTrainer

class ProposedValTrainer(RGBTDetectionTrainer):
    fusion_mode = "vqc"
    use_cbam = True

def main():
    model = YOLO("/Users/test/Q-WildFusion/runs/detect/yolov11L_rgb_ir_cbam_stage22/weights/best.pt")
    
    # Run validation with extreme Test Time Augmentation (TTA) and high resolution
    # This computationally wrings out every drop of accuracy without retraining
    metrics = model.val(
        data="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml",
        imgsz=1280,        # Upscale resolution to find smaller objects clearly
        batch=2,           # Lower batch for MPS memory
        device="mps",
        workers=2,
        augment=True,      # TTA is the key to +3-5% AP boost
        conf=0.001,        # Lower confidence to maximize recall in mAP curve
        iou=0.6,           # NMS threshold
        split="val"
    )
    
    print("\n\n======== FINAL TTA METRICS ========")
    print(f"Overall mAP@50: {metrics.box.map50:.4f}")
    print(f"Overall mAP@50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
