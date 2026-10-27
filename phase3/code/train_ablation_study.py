import argparse
import os
import sys

from ultralytics import YOLO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from phase3.code.rgbt_trainer import RGBTDetectionTrainer


ABLATIONS = {
    "no_cbam_add": {
        "name": "abl_no_cbam_add",
        "fusion_mode": "add",
        "use_cbam": False,
    },
    "cbam_add": {
        "name": "abl_cbam_no_vqc",
        "fusion_mode": "add",
        "use_cbam": True,
    },
    "cbam_vqc": {
        "name": "abl_cbam_vqc_proposed",
        "fusion_mode": "vqc",
        "use_cbam": True,
    },
    "cbam_rgvqf": {
        "name": "abl_cbam_rgvqf_proposed",
        "fusion_mode": "rgvqf",
        "use_cbam": True,
    },
    "cbam_rgvqf_no_entangle": {
        "name": "abl_cbam_rgvqf_no_entangle",
        "fusion_mode": "rgvqf",
        "use_cbam": True,
        "use_entanglement": False,
    },
    "cbam_rgvqf_rem": {
        "name": "abl_cbam_rgvqf_rem",
        "fusion_mode": "rgvqf",
        "use_cbam": True,
        "use_reliability": True,
    },
    "cmaam_add": {
        "name": "abl_cmaam_add",
        "fusion_mode": "add",
        "attention": "cmaam",
    },
    "cmaam_rgvqf_rem": {
        "name": "abl_cmaam_rgvqf_rem",
        "fusion_mode": "rgvqf",
        "attention": "cmaam",
        "use_reliability": True,
    },
    "cmaam_rgvqf_rem_single_scale": {
        "name": "abl_mqfp_single_scale",
        "fusion_mode": "rgvqf",
        "attention": "cmaam",
        "use_reliability": True,
        "multiscale": False,
    },
    "complete": {
        "name": "abl_complete",
        "fusion_mode": "rgvqf",
        "attention": "cmaam",
        "use_reliability": True,
        "alpha_rel": 0.1,
        "beta_q": 0.1,
    },
}


# Configs added for the pre-submission runs. Kept out of ABLATIONS so existing
# drivers that iterate ABLATIONS (run_step5_retrain.py etc.) are unaffected.
EXTRA_ABLATIONS = {
    "cbam_rgvqf_classical_ctrl": {
        "name": "abl_cbam_rgvqf_classical_ctrl",
        "fusion_mode": "rgvqf",
        "use_cbam": True,
        "rgvqf_learned_encoder": True,
        "rgvqf_classical_ctrl": True,
    },
}

# Pre-flight fix flags (RGBTDetectionTrainer class attributes, all default False).
FIX_FLAGS = ("attn_channel_fix", "aux_loss_batch_scale", "rem_fused_levels_only", "seeded_data_order",
             "rgvqf_learned_encoder", "rgvqf_classical_ctrl")


def make_trainer(config):
    attrs = {
        "fusion_mode": config["fusion_mode"],
        "use_cbam": config.get("use_cbam", True),
        "attention": config.get("attention"),
        "use_entanglement": config.get("use_entanglement", True),
        "use_reliability": config.get("use_reliability", False),
        "multiscale": config.get("multiscale", True),
        "alpha_rel": config.get("alpha_rel", 0.0),
        "beta_q": config.get("beta_q", 0.0),
    }
    attrs.update({flag: config[flag] for flag in FIX_FLAGS if flag in config})
    return type("AblationRGBTTrainer", (RGBTDetectionTrainer,), attrs)


def train_one(key, args):
    config = ABLATIONS[key]
    trainer_cls = make_trainer(config)

    model = YOLO(args.model)
    return model.train(
        trainer=trainer_cls,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=config["name"],
        exist_ok=args.exist_ok,
        fraction=args.fraction,
        patience=args.patience,
        pretrained=True,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=0,
        val=True,
        plots=args.plots,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/test/Q-WildFusion/phase3/dataset_aligned/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="/Users/test/Q-WildFusion/runs/detect")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(ABLATIONS),
        default=list(ABLATIONS),
        help="Subset of ablations to run.",
    )
    args = parser.parse_args()

    for key in args.only:
        config = ABLATIONS[key]
        print(
            f"[ABLATION] {config['name']}: "
            f"fusion={config['fusion_mode']}, cbam={config.get('use_cbam', True)}, "
            f"attention={config.get('attention')}, multiscale={config.get('multiscale', True)}, "
            f"alpha_rel={config.get('alpha_rel', 0.0)}, beta_q={config.get('beta_q', 0.0)}"
        )
        train_one(key, args)


if __name__ == "__main__":
    main()
