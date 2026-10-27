import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Channel-concat the two modality feature maps, reduced by a 1x1 conv."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, rgb, ir):
        return self.reduce(torch.cat([rgb, ir], dim=1))
