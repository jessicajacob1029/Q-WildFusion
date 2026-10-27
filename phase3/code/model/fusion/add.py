import torch.nn as nn


class AddFusion(nn.Module):
    """Elementwise sum of the two modality feature maps. No parameters."""

    def __init__(self, channels):
        super().__init__()

    def forward(self, rgb, ir):
        return rgb + ir
