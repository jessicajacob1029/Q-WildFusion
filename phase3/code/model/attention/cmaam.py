import torch
import torch.nn as nn


class CMAAM(nn.Module):
    """
    Cross-Modal Adaptive Attention Module (advisor Step 1).

    CBAM (cbam.py) computes each modality's channel/spatial attention from
    that modality's own pooled descriptor only. CMAAM instead builds a joint
    descriptor from BOTH modalities and derives each modality's gate from
    it, so RGB's attention is conditioned on thermal features and vice versa
    -- "each modality improves the other" per the advisor's definition.
    Replaces the rgb_cbam_pX/ir_cbam_pX pair with a single joint module.

    forward(rgb, ir) -> (rgb', ir')
    """

    def __init__(self, channels, reduction=16, channel_fix=False, fix_init_std=5e-5):
        super().__init__()
        self.channels = channels

        # -------- Cross-modal channel attention --------
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # channel_fix (default off): LeakyReLU hidden activation + small random
        # W2 init, see cbam.py -- the ReLU channel MLPs die during training.
        def act():
            return nn.LeakyReLU(0.1, inplace=True) if channel_fix else nn.ReLU(inplace=True)

        joint_channel_dim = channels * 4  # [avg_rgb, max_rgb, avg_ir, max_ir]
        hidden = max(joint_channel_dim // reduction, 8)
        self.channel_mlp_rgb = nn.Sequential(
            nn.Conv2d(joint_channel_dim, hidden, kernel_size=1, bias=False),
            act(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.channel_mlp_ir = nn.Sequential(
            nn.Conv2d(joint_channel_dim, hidden, kernel_size=1, bias=False),
            act(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # -------- Cross-modal spatial attention --------
        # joint spatial descriptor: [mean_rgb, max_rgb, mean_ir, max_ir] -> 4 channels
        self.spatial_conv_rgb = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_conv_ir = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

        # identity-stable init, matching cbam.py's convention: zeroing the
        # last layer of each gate branch makes 2*sigmoid(0) == 1.0, so CMAAM
        # starts as a no-op and any learned deviation is attributable to
        # training, not initialization luck.
        with torch.no_grad():
            for mlp in (self.channel_mlp_rgb, self.channel_mlp_ir):
                if channel_fix:
                    nn.init.normal_(mlp[2].weight, std=fix_init_std)
                else:
                    nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(self.spatial_conv_rgb.weight)
            nn.init.zeros_(self.spatial_conv_ir.weight)

    def forward(self, rgb, ir):
        # ---- Cross-modal channel attention ----
        joint_channel = torch.cat(
            [self.avg_pool(rgb), self.max_pool(rgb), self.avg_pool(ir), self.max_pool(ir)], dim=1
        )  # [B, 4C, 1, 1]

        channel_att_rgb = 2.0 * self.sigmoid(self.channel_mlp_rgb(joint_channel))
        channel_att_ir = 2.0 * self.sigmoid(self.channel_mlp_ir(joint_channel))

        rgb = rgb * channel_att_rgb
        ir = ir * channel_att_ir

        # ---- Cross-modal spatial attention ----
        joint_spatial = torch.cat(
            [
                torch.mean(rgb, dim=1, keepdim=True), torch.max(rgb, dim=1, keepdim=True)[0],
                torch.mean(ir, dim=1, keepdim=True), torch.max(ir, dim=1, keepdim=True)[0],
            ],
            dim=1,
        )  # [B, 4, H, W]

        spatial_att_rgb = 2.0 * self.sigmoid(self.spatial_conv_rgb(joint_spatial))
        spatial_att_ir = 2.0 * self.sigmoid(self.spatial_conv_ir(joint_spatial))

        rgb = rgb * spatial_att_rgb
        ir = ir * spatial_att_ir

        return rgb, ir
