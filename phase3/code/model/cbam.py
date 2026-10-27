import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Applies Channel Attention + Spatial Attention
    """

    def __init__(self, channels, reduction=16, channel_fix=False, fix_init_std=5e-5):
        super().__init__()

        # -------- Channel Attention --------
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)

        # channel_fix (default off): LeakyReLU instead of ReLU, so the hidden
        # layer cannot die (with zero-init W2 + no bias, the ReLU hidden units
        # drifted to all-negative pre-activations in trained checkpoints).
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True) if channel_fix else nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        # -------- Spatial Attention --------
        self.spatialConv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # identity-stable initialization
        # (channel_fix: W2 ~ N(0, fix_init_std) instead of zeros, so W1 gets
        # gradient from step 0 while the gate still starts at ~1.0)
        with torch.no_grad():
            if channel_fix:
                nn.init.normal_(self.mlp[2].weight, std=fix_init_std)
            else:
                nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.spatialConv.weight)

    def forward(self, x):
        # ----- Channel Attention -----
        avgOut = self.mlp(self.avgPool(x))
        maxOut = self.mlp(self.maxPool(x))
        channelAtt = 2.0 * self.sigmoid(avgOut + maxOut)
        x = x * channelAtt

        # ----- Spatial Attention -----
        avgOut = torch.mean(x, dim=1, keepdim=True)
        maxOut, _ = torch.max(x, dim=1, keepdim=True)
        spatialAtt = 2.0 * self.sigmoid(self.spatialConv(
            torch.cat([avgOut, maxOut], dim=1)
        ))

        x = x * spatialAtt
        return x


if __name__ == "__main__":
    import torch
    cbam = CBAM(channels=256)
    x = torch.randn(1, 256, 40, 40)
    y = cbam(x)
    print("CBAM output shape:", y.shape)
