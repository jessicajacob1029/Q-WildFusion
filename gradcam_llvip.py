from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/yolov11L_rgb_ir_cbam_highAP/weights/best.pt")

image_path = "phase3/dataset_aligned/images/rgb/val/150096.jpg"

results = model(image_path)

img = cv2.imread(image_path)

for r in results:
    for box in r.boxes.xyxy:
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite("gradcam_output.jpg",img)

print("Grad-CAM style visualization saved.")
