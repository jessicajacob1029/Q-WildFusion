import os
from copy import copy

import cv2
import torch
from ultralytics.data import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from phase3.code.model.yolov11_rgbt import RGBTDetectionModel


def seed_loader_shuffle(loader, seed):
    """
    B5: Ultralytics' build_dataloader seeds the DataLoader generator with a
    constant (ultralytics/data/build.py: 6148914691236517205 + RANK), so the
    train shuffle order is identical for every args.seed. The RandomSampler
    holds a reference to loader.generator, so re-seeding it here makes the
    data order depend on the run seed.
    """
    loader.generator.manual_seed(int(seed))
    return loader


class RGBTBatch(tuple):
    @property
    def shape(self):
        return self[0].shape


class RGBTDetectionValidator(DetectionValidator):
    """
    DetectionValidator whose preprocess() reproduces the same (RGB, IR)
    tuple-ification RGBTDetectionTrainer.preprocess_batch() does at train
    time, using the identical trainer helpers (_resolve_ir_root /
    _matching_ir_path) rather than a re-implementation, so batch["img"]
    reaching the model at val time is structurally identical to train time.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, trainer=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self._rgbt_trainer = trainer

    def preprocess(self, batch):
        batch = super().preprocess(batch)

        rgb = batch["img"]
        ir_root = self._rgbt_trainer._resolve_ir_root()
        ir_images = []

        for rgb_path in batch["im_file"]:
            ir_path = self._rgbt_trainer._matching_ir_path(str(rgb_path), ir_root)
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                raise FileNotFoundError(f"Missing thermal image: {ir_path}")

            ir = cv2.resize(ir, (rgb.shape[-1], rgb.shape[-2]), interpolation=cv2.INTER_LINEAR)
            ir = torch.from_numpy(ir).to(rgb.device, non_blocking=self.device.type == "cuda")
            ir = ir.float().unsqueeze(0) / 255.0
            ir_images.append(ir.repeat(3, 1, 1))

        paired = RGBTBatch((rgb, torch.stack(ir_images, dim=0)))
        assert isinstance(paired, tuple) and len(paired) == 2, (
            "val batch must be a 2-tuple (rgb, ir), matching what RGBTDetectionModel.forward() "
            "dispatches to predict_rgbt() at train time"
        )
        assert paired[0].shape == paired[1].shape, (
            f"val rgb/ir shape mismatch: {paired[0].shape} vs {paired[1].shape} "
            "(train-time preprocess_batch always produces matching shapes)"
        )
        batch["img"] = paired
        return batch


class RGBTDetectionTrainer(DetectionTrainer):
    """
    Ultralytics trainer that replaces the RGB image tensor with an
    (RGB, thermal) tuple before the model/loss forward pass.
    """

    fusion_mode = "vqc"
    use_cbam = True
    attention = None  # None derives from use_cbam; set to "none"/"cbam"/"cmaam" to override
    use_entanglement = True  # rgvqf only: False reproduces the no-entangle ablation
    use_reliability = False  # rgvqf only: REM-weighted encoding vs. fixed 0.5/0.5
    multiscale = True  # MQFP vs. single-scale (P5-only) ablation
    alpha_rel = 0.0  # composite loss weight on L_rel; 0 reproduces detection-only
    beta_q = 0.0  # composite loss weight on L_q; 0 reproduces detection-only
    # Pre-flight fixes, all default off (off reproduces the existing grid exactly):
    attn_channel_fix = False  # CBAM/CMAAM channel MLP: LeakyReLU + small random W2 init
    aux_loss_batch_scale = False  # alpha*L_rel, beta*L_q scaled by batch size like L_det
    rem_fused_levels_only = False  # don't build REM at levels with non-rgvqf fusion
    seeded_data_order = False  # B5: train shuffle order seeded from args.seed
    rgvqf_learned_encoder = False  # B6: Linear(2C,8)+BN encoder instead of channel-group pooling
    rgvqf_classical_ctrl = False  # B7: QNode replaced by tanh(Linear(8,8)) (requires B6)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = RGBTDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            fusion_mode=self.fusion_mode,
            use_cbam=self.use_cbam,
            attention=self.attention,
            use_entanglement=self.use_entanglement,
            use_reliability=self.use_reliability,
            multiscale=self.multiscale,
            alpha_rel=self.alpha_rel,
            beta_q=self.beta_q,
            verbose=verbose and RANK == -1,
            attn_channel_fix=self.attn_channel_fix,
            aux_loss_batch_scale=self.aux_loss_batch_scale,
            rem_fused_levels_only=self.rem_fused_levels_only,
            rgvqf_learned_encoder=self.rgvqf_learned_encoder,
            rgvqf_classical_ctrl=self.rgvqf_classical_ctrl,
        )
        if weights:
            model.load(weights)
            model.initialize_fusion_backbone_from_yolo()
        if RANK in {-1, 0}:
            print(f"[params] {model.param_report()}")
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Val dataset with pad=0.0 instead of Ultralytics' hard-coded 0.5
        (ultralytics/data/build.py::build_yolo_dataset). With pad=0.5 the val
        RGB is letterboxed to 800x992 with a 16 px gray border while the IR is
        cv2.resize'd to fill the whole tensor, misregistering the modalities.
        With pad=0.0, 1280x1024 LLVIP frames map to 768x960 with no border,
        identical to train geometry. Otherwise mirrors build_yolo_dataset.
        """
        if mode == "train":
            return super().build_dataset(img_path, mode, batch)
        return YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if self.seeded_data_order and mode == "train":
            seed_loader_shuffle(loader, self.args.seed)
        return loader

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return RGBTDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks, trainer=self
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        rgb = batch["img"]
        ir_root = self._resolve_ir_root()
        ir_images = []

        for rgb_path in batch["im_file"]:
            ir_path = self._matching_ir_path(str(rgb_path), ir_root)
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                raise FileNotFoundError(f"Missing thermal image: {ir_path}")

            ir = cv2.resize(ir, (rgb.shape[-1], rgb.shape[-2]), interpolation=cv2.INTER_LINEAR)
            ir = torch.from_numpy(ir).to(rgb.device, non_blocking=self.device.type == "cuda")
            ir = ir.float().unsqueeze(0) / 255.0
            ir_images.append(ir.repeat(3, 1, 1))

        batch["img"] = RGBTBatch((rgb, torch.stack(ir_images, dim=0)))
        return batch

    def _resolve_ir_root(self):
        ir_root = self.data.get("ir") or self.data.get("thermal")
        if not ir_root:
            raise KeyError("Dataset yaml must define an 'ir' path, e.g. ir: images/ir")

        if os.path.isabs(str(ir_root)):
            return str(ir_root)

        data_root = self.data.get("path") or os.path.dirname(str(self.args.data))
        return os.path.abspath(os.path.join(str(data_root), str(ir_root)))

    @staticmethod
    def _matching_ir_path(rgb_path, ir_root):
        normalized = rgb_path.replace("\\", "/")
        if "/train/" in normalized:
            split = "train"
        elif "/val/" in normalized:
            split = "val"
        else:
            split = os.path.basename(os.path.dirname(rgb_path))

        return os.path.join(ir_root, split, os.path.basename(rgb_path))
