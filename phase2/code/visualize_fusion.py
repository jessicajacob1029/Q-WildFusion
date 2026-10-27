import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import RGBThermalDataset
from train_model import FusionYOLO

DEVICE = "cpu"
SAVE_DIR = "phase2/visuals"
CHECKPOINT = "phase2_fusion_epoch0_batch50.pt"  # change if needed

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
dataset = RGBThermalDataset(
    root="phase2/dataset",
    split="val"
)

# -----------------------
# Load model
# -----------------------
model = FusionYOLO().to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

print("✅ Model loaded for visualization")

# -----------------------
# Feature → heatmap
# -----------------------
def feature_to_heatmap(feat):
    feat = feat.detach().cpu().numpy()
    heatmap = feat.mean(axis=0)
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-6)
    return heatmap

# -----------------------
# Visualize samples
# -----------------------
for idx in range(5):
    rgb, ir, _ = dataset[idx]

    rgb_t = rgb.unsqueeze(0).to(DEVICE)
    ir_t = ir.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fused_feat = model.forward_features(rgb_t, ir_t)

    fused_feat = fused_feat[0]  # [C,H,W]
    heatmap = feature_to_heatmap(fused_feat)

    rgb_img = rgb.permute(1, 2, 0).cpu().numpy()
    ir_img = ir.squeeze().cpu().numpy()

    # Plot safely using matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(rgb_img)
    axs[0].set_title("RGB")
    axs[0].axis("off")

    axs[1].imshow(ir_img, cmap="gray")
    axs[1].set_title("IR")
    axs[1].axis("off")

    axs[2].imshow(heatmap, cmap="jet")
    axs[2].set_title("Fused Heatmap")
    axs[2].axis("off")

    save_path = f"{SAVE_DIR}/sample_{idx}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"💾 Saved {save_path}")

print("🎉 Fusion visualization complete")
