import copy

import torch
import torch.nn as nn

from .attention import CMAAM
from .cbam import CBAM
from .fusion import build_fusion
from .reliability import REM
from ..losses.composite import reliability_consistency_loss, reliability_target

ATTENTION_MODES = ("none", "cbam", "cmaam")

# REM's (w_rgb, w_th) only means anything to fusion classes that accept
# reliability weights in their forward() signature.
RELIABILITY_AWARE_FUSION_MODES = ("rgvqf",)

# Stage 5 (MQFP) single-scale ablation: when multiscale=False, P3/P4 fall
# back to this fusion mode instead of the requested one, isolating whether
# spreading the requested fusion across all three scales (vs. only P5) helps.
SINGLE_SCALE_FALLBACK_MODE = "add"


class RGBTBackboneCBAM(nn.Module):
    """
    Backbone-level RGB-Thermal fusion with attention.

    Pipeline:
    RGB/Thermal image -> dual backbones -> attention -> [REM] -> fusion.
    Attention is either per-modality (cbam: each modality only sees its own
    features) or cross-modal (cmaam: each modality's gate is conditioned on
    the other, see attention/cmaam.py).
    Supports fusion ablation: add | concat | vqc (alias: qinspired) | rgvqf
    Supports attention ablation: none | cbam | cmaam
    """

    def __init__(self, fusion_mode="vqc", use_cbam=True, attention=None, use_entanglement=True,
                 use_reliability=False, multiscale=True, source_backbone=None, attn_channel_fix=False,
                 rem_fused_levels_only=False, rgvqf_learned_encoder=False, rgvqf_classical_ctrl=False):
        super().__init__()

        self.fusion_mode = fusion_mode  # the requested mode; see fusion_mode_p3/p4/p5 for what's actually built
        # Default False: REM is new (Stage 4) and opt-in, so existing callers
        # that don't pass this flag keep today's unweighted (0.5/0.5) fusion.
        self.use_reliability = use_reliability
        self.last_reliability_weights = None  # populated by forward() for logging
        self.last_l_rel = None  # populated by forward() for the composite loss (Stage 6)

        # MQFP (Stage 5): multiscale=True applies fusion_mode at all three
        # levels (the plan's default pipeline). multiscale=False is the
        # single-scale ablation -- P3/P4 fall back to a plain classical
        # fusion so only P5 uses the requested (typically rgvqf) mode.
        self.multiscale = multiscale
        self.fusion_mode_p3 = fusion_mode if multiscale else SINGLE_SCALE_FALLBACK_MODE
        self.fusion_mode_p4 = fusion_mode if multiscale else SINGLE_SCALE_FALLBACK_MODE
        self.fusion_mode_p5 = fusion_mode

        # `attention` is the new {none, cbam, cmaam} flag. `use_cbam` is kept
        # for backward compatibility with existing callers (rgbt_trainer.py,
        # yolov11_rgbt.py, train_ablation_study.py); when `attention` isn't
        # given explicitly, it's derived from `use_cbam`.
        if attention is None:
            attention = "cbam" if use_cbam else "none"
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.attention = attention
        self.use_cbam = attention == "cbam"

        if source_backbone is None:
            raise ValueError("source_backbone is required to match the selected YOLO scale")

        source_backbone = list(source_backbone)
        self.rgb_backbone = nn.ModuleList(copy.deepcopy(source_backbone))
        self.ir_backbone = nn.ModuleList(copy.deepcopy(source_backbone))

        p3_channels = self._out_channels(source_backbone[4])
        p4_channels = self._out_channels(source_backbone[6])
        p5_channels = self._out_channels(source_backbone[10])

        # Fusion modules, one per pyramid level, built from the registry.
        # `use_entanglement` is only meaningful for rgvqf (the no-entangle
        # ablation); other fusion classes don't accept it, so it's only
        # forwarded when relevant. Built from the per-level mode (see MQFP
        # note above), not the raw requested `fusion_mode`.
        def _kwargs(mode):
            if mode != "rgvqf":
                return {}
            kw = {"entangle": use_entanglement}
            if rgvqf_learned_encoder:  # only passed when on, so flag-off calls are unchanged
                kw["learned_encoder"] = True
            if rgvqf_classical_ctrl:
                kw["classical_ctrl"] = True
            return kw

        self.fuse_p3 = build_fusion(self.fusion_mode_p3, p3_channels, **_kwargs(self.fusion_mode_p3))
        self.fuse_p4 = build_fusion(self.fusion_mode_p4, p4_channels, **_kwargs(self.fusion_mode_p4))
        self.fuse_p5 = build_fusion(self.fusion_mode_p5, p5_channels, **_kwargs(self.fusion_mode_p5))

        # Attention per modality before fusion
        # attn_channel_fix (default off): non-dying channel MLP, see cbam.py / cmaam.py
        fix = {"channel_fix": True} if attn_channel_fix else {}
        if self.attention == "cbam":
            self.rgb_cbam_p3 = CBAM(p3_channels, **fix)
            self.rgb_cbam_p4 = CBAM(p4_channels, **fix)
            self.rgb_cbam_p5 = CBAM(p5_channels, **fix)
            self.ir_cbam_p3 = CBAM(p3_channels, **fix)
            self.ir_cbam_p4 = CBAM(p4_channels, **fix)
            self.ir_cbam_p5 = CBAM(p5_channels, **fix)
        elif self.attention == "cmaam":
            self.cmaam_p3 = CMAAM(p3_channels, **fix)
            self.cmaam_p4 = CMAAM(p4_channels, **fix)
            self.cmaam_p5 = CMAAM(p5_channels, **fix)

        # Reliability Estimation Module, one per pyramid level (mirrors the
        # per-level fusion/attention pattern). Only meaningful feeding into
        # rgvqf; harmless (but wasted) to enable otherwise.
        # rem_fused_levels_only (default off): skip REM at levels whose fusion
        # mode can't consume reliability weights (P3/P4 in the single-scale
        # ablation), where it would be built but never called.
        if self.use_reliability:
            for lvl, mode in (("p3", self.fusion_mode_p3), ("p4", self.fusion_mode_p4), ("p5", self.fusion_mode_p5)):
                if not rem_fused_levels_only or mode in RELIABILITY_AWARE_FUSION_MODES:
                    setattr(self, f"rem_{lvl}", REM())

    @staticmethod
    def _out_channels(module):
        if hasattr(module, "conv"):
            return module.conv.out_channels
        if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
            return module.cv2.conv.out_channels
        raise AttributeError(f"Cannot infer output channels for {type(module).__name__}")

    @staticmethod
    def _forward_backbone(layers, x):
        p3 = p4 = p5 = None
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx == 4:
                p3 = x
            elif idx == 6:
                p4 = x
            elif idx == 10:
                p5 = x
        return p3, p4, p5

    # ---------------------------------
    # Forward
    # ---------------------------------
    def forward(self, rgb, ir):
        # ---- Dual streams ----
        p3_rgb, p4_rgb, p5_rgb = self._forward_backbone(self.rgb_backbone, rgb)
        p3_ir, p4_ir, p5_ir = self._forward_backbone(self.ir_backbone, ir)

        # ---- Attention refinement per modality ----
        if self.attention == "cbam":
            p3_rgb = self.rgb_cbam_p3(p3_rgb)
            p4_rgb = self.rgb_cbam_p4(p4_rgb)
            p5_rgb = self.rgb_cbam_p5(p5_rgb)
            p3_ir = self.ir_cbam_p3(p3_ir)
            p4_ir = self.ir_cbam_p4(p4_ir)
            p5_ir = self.ir_cbam_p5(p5_ir)
        elif self.attention == "cmaam":
            p3_rgb, p3_ir = self.cmaam_p3(p3_rgb, p3_ir)
            p4_rgb, p4_ir = self.cmaam_p4(p4_rgb, p4_ir)
            p5_rgb, p5_ir = self.cmaam_p5(p5_rgb, p5_ir)
        # "none" passes features through unchanged.

        # ---- Reliability estimation (post-attention) + fusion, per level ----
        # Decided per level (not from the single top-level fusion_mode)
        # because MQFP's single-scale ablation (multiscale=False) can give
        # P3/P4 a fallback fusion mode that doesn't accept reliability
        # weights at all, even when P5 is rgvqf.
        levels = [
            ("p3", self.fuse_p3, getattr(self, "rem_p3", None), self.fusion_mode_p3, p3_rgb, p3_ir),
            ("p4", self.fuse_p4, getattr(self, "rem_p4", None), self.fusion_mode_p4, p4_rgb, p4_ir),
            ("p5", self.fuse_p5, getattr(self, "rem_p5", None), self.fusion_mode_p5, p5_rgb, p5_ir),
        ]

        outputs = {}
        reliability_weights = {}
        l_rel_terms = []
        for name, fuse, rem, mode, feat_rgb, feat_ir in levels:
            use_rem = self.use_reliability and mode in RELIABILITY_AWARE_FUSION_MODES
            if use_rem:
                w_rgb, w_th, weights = rem(feat_rgb, feat_ir)
                reliability_weights[name] = weights.detach()
                target = reliability_target(feat_rgb, feat_ir)
                l_rel_terms.append(reliability_consistency_loss(weights, target))
                outputs[name] = fuse(feat_rgb, feat_ir, w_rgb=w_rgb, w_th=w_th)
            else:
                outputs[name] = fuse(feat_rgb, feat_ir)

        self.last_reliability_weights = reliability_weights or None
        self.last_l_rel = torch.stack(l_rel_terms).mean() if l_rel_terms else None

        return [outputs["p3"], outputs["p4"], outputs["p5"]]
