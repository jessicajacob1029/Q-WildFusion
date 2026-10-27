"""
Single-stream baseline trainer for Phase 3: stock Ultralytics DetectionTrainer
(stock model, loss, validator) with only two changes, so baselines are measured
exactly like the fusion models:
  - val dataset built with pad=0.0 (768x960, no border), as RGBTDetectionTrainer.build_dataset
  - seeded_data_order (B5): train shuffle generator seeded from args.seed
"""
from ultralytics.data import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from phase3.code.rgbt_trainer import seed_loader_shuffle


class BaselineTrainer(DetectionTrainer):
    seeded_data_order = True
    # False (thermal-only baseline): drop the Albumentations photometric step (Blur/MedianBlur/ToGray/CLAHE,
    # p=0.01 each) from the train transforms, mirroring the fusion IR stream, which gets no photometric
    # augmentation. HSV is switched off via hsv_h/s/v=0 in the run args.
    photometric_aug = True

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            ds = super().build_dataset(img_path, mode, batch)
            if not self.photometric_aug:
                ds.transforms.transforms = [t for t in ds.transforms.transforms if type(t).__name__ != "Albumentations"]
            return ds
        return YOLODataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=False, hyp=self.args,
                           rect=True, cache=self.args.cache or None, single_cls=self.args.single_cls or False,
                           stride=max(int(unwrap_model(self.model).stride.max()), 32), pad=0.0,
                           prefix=colorstr(f"{mode}: "), task=self.args.task, classes=self.args.classes,
                           data=self.data, fraction=1.0)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if self.seeded_data_order and mode == "train":
            seed_loader_shuffle(loader, self.args.seed)
        return loader
