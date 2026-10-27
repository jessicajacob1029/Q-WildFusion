import torch
import torch.nn as nn
from ultralytics import YOLO

class FusionYOLO(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------------
        # Load YOLOv8 backbone
        # -----------------------
        yolo = YOLO("yolov8n.pt")
        base_model = yolo.model

        # Backbone layers (shared architecture)
        self.rgb_backbone = base_model.model[:10]
        self.ir_backbone = base_model.model[:10]

        # Channel sizes at P3 level for YOLOv8n
        # RGB: 256, IR: 256 → concat = 512
        self.channel_reduce = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Detection head (kept for Phase-3)
        self.detect = base_model.model[10]

    # -----------------------
    # Backbone forward passes
    # -----------------------
    def forward_rgb(self, x):
        for layer in self.rgb_backbone:
            x = layer(x)
        return x

    def forward_ir(self, x):
        for layer in self.ir_backbone:
            x = layer(x)
        return x

    # -----------------------
    # 🔍 Feature-only forward (FOR VISUALIZATION)
    # -----------------------
    def forward_features(self, rgb, ir):
        """
        Returns fused feature map only
        Shape: [B, 256, H, W]
        """
        f_rgb = self.forward_rgb(rgb)
        f_ir = self.forward_ir(ir)

        fused = torch.cat([f_rgb, f_ir], dim=1)
        fused = self.channel_reduce(fused)

        return fused

    # -----------------------
    # 🔁 Training forward (Phase-2)
    # -----------------------
    def forward(self, rgb, ir, targets=None):
        """
        Phase-2 training forward
        """

        fused = self.forward_features(rgb, ir)

        # Fake detection output list (to match YOLO-style outputs)
        preds = [fused]

        # Proxy loss for fusion pretraining
        if targets is not None:
            loss = fused.pow(2).mean()
            return preds, loss, None

        return preds
