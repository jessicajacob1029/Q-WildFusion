from ultralytics.nn.tasks import DetectionModel
from model.cbam import CBAM
import torch
import cv2
import os


class RGBIR_YOLOv11_CBAM(DetectionModel):
    """
    YOLOv11 + CBAM + RGB–IR dual-input model (Ultralytics-safe)
    """

    def __init__(self, cfg="yolo11s.yaml", ch=3, nc=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc)

        # CBAM modules (created lazily AFTER warmup)
        self._cbamRGB = None
        self._cbamIR = None

        self.ir_root = None

    def _is_warmup(self):
        """
        Detect Ultralytics internal warmup forward
        """
        return not hasattr(self, "stride") or self.stride is None

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        if self.ir_root is None:
            self.ir_root = self.args.data.get("ir")

        rgb_imgs = batch["img"]
        paths = batch["im_file"]

        ir_imgs = []
        for p in paths:
            fname = os.path.basename(p)
            split = "train" if "train" in p else "val"
            ir_path = os.path.join(self.ir_root, split, fname)

            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                raise FileNotFoundError(f"Missing IR image: {ir_path}")

            ir = cv2.resize(ir, (rgb_imgs.shape[-1], rgb_imgs.shape[-2]))
            ir = torch.from_numpy(ir).unsqueeze(0)
            ir_imgs.append(ir)

        ir_imgs = torch.stack(ir_imgs).float().to(rgb_imgs.device) / 255.0
        batch["img"] = (rgb_imgs, ir_imgs)

        return batch

    def forward(self, x):
        # 🚨 Skip CBAM during Ultralytics warmup forward
        use_cbam = not self._is_warmup()

        if isinstance(x, tuple):
            rgb, ir = x
        else:
            rgb, ir = x, None

        # ---------- RGB backbone ----------
        rgb_feats = []
        xi = rgb

        for i, m in enumerate(self.model):
            xi = m(xi)

            if use_cbam and i == 6:
                if self._cbamRGB is None:
                    self._cbamRGB = CBAM(xi.shape[1]).to(xi.device)
                xi = self._cbamRGB(xi)

            rgb_feats.append(xi)

        # ---------- IR backbone ----------
        if ir is not None:
            ir_feats = []
            xi = ir

            for i, m in enumerate(self.model):
                xi = m(xi)

                if use_cbam and i == 6:
                    if self._cbamIR is None:
                        self._cbamIR = CBAM(xi.shape[1]).to(xi.device)
                    xi = self._cbamIR(xi)

                ir_feats.append(xi)

            fused = [r + t for r, t in zip(rgb_feats, ir_feats)]
        else:
            fused = rgb_feats

        return self.detect(fused)
