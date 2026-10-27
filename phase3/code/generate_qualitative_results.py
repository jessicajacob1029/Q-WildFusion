import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def generate_qualitative_results(model_path, images_dir, output_dir, conf=0.25):
    """
    Generate qualitative results with correct 'human' labels
    """

    # Load model
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))

    print(f"Found {len(image_files)} images to process")

    processed = 0
    for img_path in image_files:
        try:
            # Run inference
            results = model(str(img_path), conf=conf, verbose=False)

            if len(results) > 0 and results[0].boxes is not None:
                # Override class names to 'human'
                results[0].names = {0: 'human'}

                # Save result with bounding boxes
                result_img = results[0].plot(labels=True, conf=True)

                # Save with same filename
                output_path = os.path.join(output_dir, img_path.name)
                cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(image_files)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"✅ Generated {processed} qualitative results with 'human' labels in {output_dir}")

if __name__ == "__main__":
    # Use the best model from the latest training run
    model_path = "runs/detect/yolov11L_rgb_ir_cbam_highAP/weights/best.pt"

    # Input images (validation set)
    images_dir = "phase3/dataset_aligned/images/rgb/val"

    # Output directory for new qualitative results
    output_dir = "runs/detect/qualitative_results_human"

    if os.path.exists(model_path):
        generate_qualitative_results(model_path, images_dir, output_dir)
    else:
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for pt_file in Path("runs/detect").glob("**/weights/best.pt"):
            print(f"  - {pt_file}")