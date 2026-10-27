from ultralytics import YOLO

model = YOLO("runs/detect/yolov11L_rgb_ir_cbam_highAP/weights/best.pt")

# change displayed class name
model.model.names = {0: "human"}

model.predict(
    source="phase3/dataset_aligned/images/rgb/val",
    save=True,
    conf=0.25
)

print("Predictions generated with label 'human'")
