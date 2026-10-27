import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/detect/abl_vqc_real/results.csv")

plt.figure(figsize=(8,5))
plt.plot(df["metrics/mAP50(B)"], label="mAP@50")
plt.plot(df["metrics/mAP50-95(B)"], label="mAP@50–95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.legend()
plt.title("Detection Performance over Epochs")
plt.grid()
plt.show()
