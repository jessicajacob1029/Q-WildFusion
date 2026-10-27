import torch
import torch.nn as nn


class QInspiredFusion(nn.Module):
    """
    Quantum-inspired (classical) gating fusion for RGB-Thermal feature maps.

    Formerly `VQCFusion` in `model/vqc_fusion.py`. Renamed to make explicit
    that this is a sin/cos-parameterized classical gate, not a variational
    quantum circuit — see `rgvqf.py` for the real PennyLane-backed fusion.

    The architecture diagram places GAP + linear projection before the VQC
    block. For detection, we still need to return a spatial feature map for
    the YOLO neck, so this branch produces channel gates from pooled
    descriptors and applies them to a learned spatial fusion of the two
    modality maps.
    """

    def __init__(self, channels, latent_dim=8):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(channels * 2, latent_dim)

        self.theta = nn.Parameter(torch.randn(latent_dim))
        self.phi = nn.Parameter(torch.randn(latent_dim))

        self.expand = nn.Linear(latent_dim, channels)
        self.spatial_mix = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # identity-stable initialization (averaging)
        with torch.no_grad():
            self.spatial_mix.weight.zero_()
            for i in range(channels):
                self.spatial_mix.weight[i, i, 0, 0] = 1.0
                self.spatial_mix.weight[i, i + channels, 0, 0] = 1.0
            if self.spatial_mix.bias is not None:
                self.spatial_mix.bias.zero_()

            self.expand.weight.zero_()
            if self.expand.bias is not None:
                self.expand.bias.zero_()

    def forward(self, fr, ft):
        # GAP + linear projection before the quantum-inspired block.
        qr = self.gap(fr).flatten(1)
        qt = self.gap(ft).flatten(1)
        q = self.project(torch.cat([qr, qt], dim=1))

        # Quantum-inspired rotations.
        q = torch.sin(q * self.theta.view(1, -1)) + \
            torch.cos(q * self.phi.view(1, -1))

        # Entanglement-like mixing
        q = q + torch.roll(q, shifts=1, dims=1)

        gate = self.sigmoid(self.expand(q)).view(fr.shape[0], fr.shape[1], 1, 1)
        fused = self.spatial_mix(torch.cat([fr, ft], dim=1))

        return fused * gate
