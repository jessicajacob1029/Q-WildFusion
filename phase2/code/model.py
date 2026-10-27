import torch
import torch.nn as nn
from ultralytics import YOLO


class SimpleFusionYOLO(nn.Module):
    """
    RGB + Thermal fusion module for YOLOv8.
    This model:
    - uses YOLOv8 backbone+neck unchanged
    - extracts the final feature map (before Detect)
    - fuses RGB and IR features
    - returns fused features (NOT detection yet)
    """

    def __init__(self, yolo_weights="yolov8n.pt"):
        super().__init__()

        # Load YOLOv8 detection model
        self.yolo = YOLO(yolo_weights).model

        # Placeholder to store features
        self._feature = None

        # Hook: capture output of the layer BEFORE Detect
        def hook_fn(module, input, output):
            self._feature = output  # shape: [B, 256, H, W]

        # Register hook on neck output
        self.yolo.model[-2].register_forward_hook(hook_fn)

        # Fusion layer: concat(RGB, IR) → reduce channels
        self.fusion = nn.Conv2d(
            in_channels=256 * 2,
            out_channels=256,
            kernel_size=1,
            bias=False
        )

    def extract_feature(self, x):
        """
        Run YOLO forward to extract last feature map
        """
        self._feature = None
        _ = self.yolo(x)
        return self._feature

    def forward(self, rgb, ir):
        """
        Forward pass:
        - extract RGB features
        - extract IR features
        - fuse them
        - return fused feature map
        """

        f_rgb = self.extract_feature(rgb)
        f_ir = self.extract_feature(ir)

        # Feature-level fusion
        fused = torch.cat([f_rgb, f_ir], dim=1)
        fused = self.fusion(fused)

        return fused
