import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO

# Must import RGBTDetectionTrainer so ultralytics can find the class
from phase3.code.rgbt_trainer import RGBTDetectionTrainer

class ProposedValTrainer(RGBTDetectionTrainer):
    fusion_mode = "vqc"
    use_cbam = True

def main():
    model = YOLO("/Users/test/Q-WildFusion/runs/detect/yolov11L_rgb_ir_cbam_stage22/weights/best.pt")
    
    # Run validation
    metrics = model.val(
        data="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml",
        imgsz=960,
        batch=4,
        device="mps",
        workers=4,
        split="val",
        save_json=True
    )
    
if __name__ == "__main__":
    main()
