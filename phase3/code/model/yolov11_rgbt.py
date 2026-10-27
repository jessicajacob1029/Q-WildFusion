import torch

from ultralytics.nn.tasks import DetectionModel

from .rgbt_backbone_cbam import RGBTBackboneCBAM
from ..losses.composite import quantum_regularization_loss


class RGBTDetectionModel(DetectionModel):
    """
    YOLOv11 detection model with the proposed RGB-Thermal front-end:

    RGB image      -> RGB backbone      -> CBAM
                                                   -> VQC fusion -> YOLO neck/head
    Thermal image  -> Thermal backbone  -> CBAM
    """

    def __init__(self, cfg="yolo11s.yaml", ch=3, nc=None, fusion_mode="vqc", use_cbam=True,
                 attention=None, use_entanglement=True, use_reliability=False, multiscale=True,
                 alpha_rel=0.0, beta_q=0.0, verbose=True, attn_channel_fix=False,
                 aux_loss_batch_scale=False, rem_fused_levels_only=False, rgvqf_learned_encoder=False,
                 rgvqf_classical_ctrl=False):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.fusion_mode = fusion_mode
        self.use_cbam = use_cbam
        # Composite loss weights (Stage 6): L = L_det + alpha_rel*L_rel + beta_q*L_q.
        # Both default to 0, which exactly reproduces the detection-only loss.
        self.alpha_rel = alpha_rel
        self.beta_q = beta_q
        # aux_loss_batch_scale (default off): v8DetectionLoss returns L_det x batch
        # size, so unscaled alpha/beta act as alpha/B, beta/B relative to the
        # per-image detection loss. On: the aux terms get the same x B.
        self.aux_loss_batch_scale = aux_loss_batch_scale
        self.fusion_backbone = RGBTBackboneCBAM(
            fusion_mode=fusion_mode,
            use_cbam=use_cbam,
            attention=attention,
            use_entanglement=use_entanglement,
            use_reliability=use_reliability,
            multiscale=multiscale,
            source_backbone=self.model[:11],
            attn_channel_fix=attn_channel_fix,
            rem_fused_levels_only=rem_fused_levels_only,
            rgvqf_learned_encoder=rgvqf_learned_encoder,
            rgvqf_classical_ctrl=rgvqf_classical_ctrl,
        )

    def initialize_fusion_backbone_from_yolo(self):
        for source, rgb_layer, thermal_layer in zip(
            self.model[:11],
            self.fusion_backbone.rgb_backbone,
            self.fusion_backbone.ir_backbone,
        ):
            rgb_layer.load_state_dict(source.state_dict())
            thermal_layer.load_state_dict(source.state_dict())

    def param_report(self):
        """Param counts. model.model[0:11] (the stock single-stream backbone) is
        never called on the paired (RGB, IR) path, so it is excluded from 'used'."""
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        unused = sum(count(self.model[i]) for i in range(11))
        quantum = sum(p.numel() for n, p in self.fusion_backbone.named_parameters() if n.endswith("q_weights"))
        return {
            "total": count(self),
            "unused_stock_backbone": unused,
            "used": count(self) - unused,
            "rgb_backbone": count(self.fusion_backbone.rgb_backbone),
            "ir_backbone": count(self.fusion_backbone.ir_backbone),
            "neck_head": sum(count(self.model[i]) for i in range(11, len(self.model))),
            "fusion_attention_rem": count(self.fusion_backbone) - count(self.fusion_backbone.rgb_backbone)
            - count(self.fusion_backbone.ir_backbone),
            "quantum": quantum,
        }

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        if isinstance(x, tuple):
            return self.predict_rgbt(x)
        return super().forward(x, *args, **kwargs)

    def predict_rgbt(self, x):
        rgb, thermal = x
        p3, p4, p5 = self.fusion_backbone(rgb, thermal)

        layers = self.model

        # YOLOv11 neck graph, using fused backbone outputs at the original
        # skip levels: layer 4 (P3), layer 6 (P4), and layer 10 (P5).
        x11 = layers[11](p5)
        x12 = layers[12]([x11, p4])
        x13 = layers[13](x12)

        x14 = layers[14](x13)
        x15 = layers[15]([x14, p3])
        x16 = layers[16](x15)

        x17 = layers[17](x16)
        x18 = layers[18]([x17, x13])
        x19 = layers[19](x18)

        x20 = layers[20](x19)
        x21 = layers[21]([x20, p5])
        x22 = layers[22](x21)

        return layers[23]([x16, x19, x22])

    def loss(self, batch, preds=None):
        """
        Composite loss (Stage 6): L = L_det + alpha_rel*L_rel + beta_q*L_q.

        The tensor used for `.sum().backward()` (the first return value) is
        extended with the extra terms; the tensor used for Ultralytics'
        progress-bar/CSV logging (the second return value, tied to
        DetectionTrainer's hardcoded 3-element `loss_names`) is left exactly
        as v8DetectionLoss produces it, so alpha_rel=beta_q=0 reproduces the
        detection-only run byte-for-byte and nothing about Ultralytics'
        reporting machinery needs to change. L_rel/L_q are printed directly
        instead, since they're not in the reported tensor.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])  # populates fusion_backbone side effects below

        det_loss, det_loss_items = self.criterion(preds, batch)

        l_rel = self.fusion_backbone.last_l_rel
        if l_rel is None:
            l_rel = det_loss.new_zeros(())

        q_outs = [
            m.last_q_out
            for m in (self.fusion_backbone.fuse_p3, self.fusion_backbone.fuse_p4, self.fusion_backbone.fuse_p5)
            if getattr(m, "last_q_out", None) is not None
        ]
        l_q = torch.stack([quantum_regularization_loss(q) for q in q_outs]).mean() if q_outs else det_loss.new_zeros(())

        extra = torch.stack([self.alpha_rel * l_rel, self.beta_q * l_q])
        if getattr(self, "aux_loss_batch_scale", False):  # getattr: checkpoints pickled before this flag
            img = batch["img"]
            extra = extra * (img[0] if isinstance(img, tuple) else img).shape[0]
        total_loss = torch.cat([det_loss, extra])

        if self.alpha_rel != 0 or self.beta_q != 0:
            print(
                f"[composite loss] det={det_loss.sum().item():.4f} "
                f"l_rel={l_rel.item():.4f} (x{self.alpha_rel}) "
                f"l_q={l_q.item():.4f} (x{self.beta_q})"
            )

        return total_loss, det_loss_items
