from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/abl_vqc_real/weights/best.pt")

# Input image path
image_path = "test.jpg"   # change this
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Run detection
results = model(
    "phase3/dataset_aligned/images/rgb/val",
    conf=0.25,
    save=True,
    project="outputs",
    name="demo"
)

# Save and display results
for r in results:
    annotated = r.plot()
    out_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"Saved result to {out_path}")

