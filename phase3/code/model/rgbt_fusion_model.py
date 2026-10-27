import torch
import torch.nn as nn

from model.cbam import CBAM


class RGBThermalFusion(nn.Module):
    """
    Attention-Enhanced RGB–Thermal Fusion Module
    CBAM (per modality) + VQC fusion
    """

    def __init__(self, rgbBackbone, thermalBackbone, neck, vqcFusion, channels=256):
        super().__init__()

        self.rgbBackbone = rgbBackbone
        self.thermalBackbone = thermalBackbone

        # CBAM per modality
        self.cbamRGB = CBAM(channels=channels)
        self.cbamThermal = CBAM(channels=channels)

        # Quantum fusion
        self.vqcFusion = vqcFusion

        # Optional post-fusion attention
        self.cbamFused = CBAM(channels=channels)

        # YOLO neck (PAN/FPN)
        self.neck = neck

    def forward(self, rgb, thermal):
        # Backbone feature extraction
        rgbFeat = self.rgbBackbone(rgb)
        thermalFeat = self.thermalBackbone(thermal)

        # Modality-specific attention
        rgbFeat = self.cbamRGB(rgbFeat)
        thermalFeat = self.cbamThermal(thermalFeat)

        # Hybrid quantum-classical fusion
        fusedFeat = self.vqcFusion(rgbFeat, thermalFeat)

        # Cross-modal refinement
        fusedFeat = self.cbamFused(fusedFeat)

        # YOLO neck
        out = self.neck(fusedFeat)
        return out
