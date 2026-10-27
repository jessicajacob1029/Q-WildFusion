# phase3/code/predict.py
import torch
from PIL import Image
from torchvision import transforms
from phase3.code.model import RGBIR_YOLOv8_CBAM

def load_image(path, gray=False):
    img = Image.open(path)
    img = img.convert("L" if gray else "RGB")

    tf = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    return tf(img).unsqueeze(0)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = RGBIR_YOLOv8_CBAM().to(device)
    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.eval()

    rgb = load_image("sample_rgb.jpg").to(device)
    ir  = load_image("sample_ir.jpg", gray=True).to(device)

    with torch.no_grad():
        preds = model(rgb, ir)

    print("Predictions:", preds)

if __name__ == "__main__":
    main()
