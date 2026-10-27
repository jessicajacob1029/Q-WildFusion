import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO results
df = pd.read_csv("runs/detect/abl_vqc_real/results.csv")

plt.figure(figsize=(8, 5))
plt.plot(df["metrics/mAP50(B)"], label="mAP@50")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP@50–95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("YOLOv11 RGB–Thermal Detection Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

