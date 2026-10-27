import torch
import torch.nn as nn
from ultralytics import YOLO


# ===================== CBAM MODULE ===================== #
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ================= RGB–IR YOLOv8 + CBAM ================= #
class RGBIR_YOLOv8_CBAM(nn.Module):
    """
    Ultralytics-compatible RGB–IR fusion YOLOv8 model

    Input  : (B, 4, H, W) -> RGB(3) + IR(1)
    Output : YOLOv8 Detect predictions
    """

    def __init__(self):
        super().__init__()

        # Base YOLOv8 model
        base_model = YOLO("yolov8s.yaml").model
        self.yolo = base_model

        # 🔥 REQUIRED by Ultralytics trainer
        self.yaml = base_model.yaml

        # RGB & IR channel adapters
        self.rgb_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.ir_conv = nn.Conv2d(1, 3, kernel_size=1)

        # CBAM after fusion (6 channels)
        self.cbam = CBAM(6)

    def forward(self, x):
        # x: (B, 4, H, W)
        rgb = x[:, :3, :, :]
        ir = x[:, 3:, :, :]

        rgb = self.rgb_conv(rgb)
        ir = self.ir_conv(ir)

        fused = torch.cat([rgb, ir], dim=1)  # (B, 6, H, W)
        fused = self.cbam(fused)

        # Reduce back to 3 channels for YOLO
        fused = fused[:, :3, :, :]

        return self.yolo(fused)
