import torch
from ultralytics import YOLO


def load_frozen_model(weights_path):
    model = YOLO(weights_path).model
    model.eval()  # inference mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


if __name__ == "__main__":
    model = load_frozen_model(
        "frozen_models/qwildfusion_classical_best.pt"
    )
    print("Model loaded and frozen successfully.")
