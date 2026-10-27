import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_frozen_model(weights_path):
    model = YOLO(weights_path).model
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess_image(img_path, img_size=640):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(DEVICE)


def extract_backbone_features(model, x):
    """
    Forward through backbone only.
    """
    features = []
    for layer in model.model[:10]:  # backbone layers
        x = layer(x)
    return x  # final backbone feature map


def global_average_pooling(x):
    return torch.mean(x, dim=[2, 3])  # (B, C)


def reduce_dim(x, target_dim=6):
    """
    Simple linear projection to quantum dimension.
    """
    proj = torch.nn.Linear(x.shape[1], target_dim, bias=False).to(x.device)
    return proj(x)


def main():
    weights = "frozen_models/qwildfusion_classical_best.pt"
    image_dir = "phase3/dataset_aligned/images/rgb/val"

    model = load_frozen_model(weights)

    features = []
    filenames = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        x = preprocess_image(img_path)

        with torch.no_grad():
            feat_map = extract_backbone_features(model, x)
            pooled = global_average_pooling(feat_map)
            reduced = reduce_dim(pooled, target_dim=6)

        features.append(reduced.cpu().numpy()[0])
        filenames.append(fname)

    features = np.array(features)

    os.makedirs("quantum_features", exist_ok=True)
    np.save("quantum_features/features_val.npy", features)
    np.save("quantum_features/filenames_val.npy", filenames)

    print("Feature extraction complete.")
    print("Saved features shape:", features.shape)


if __name__ == "__main__":
    main()
