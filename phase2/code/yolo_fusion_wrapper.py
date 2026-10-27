import torch
import torch.nn as nn
from ultralytics import YOLO
from phase2.code.model import SimpleFusionYOLO


class YOLOFusionWrapper(nn.Module):
    """
    Wrapper that injects RGB–Thermal fusion into YOLOv8
    """

    def __init__(self, yolo_weights="yolov8n.pt"):
        super().__init__()

        # Base YOLO model
        self.yolo = YOLO(yolo_weights).model

        # Fusion module
        self.fusion = SimpleFusionYOLO(yolo_weights)

        # Detect head
        self.detect = self.yolo.model[-1]

    def forward(self, rgb, ir):
        # Extract fused features
        fused = self.fusion(rgb, ir)

        # Let YOLO Detect handle multi-scale logic internally
        preds = self.detect([fused])

        return preds
