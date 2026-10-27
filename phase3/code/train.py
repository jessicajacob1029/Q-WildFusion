import os
import sys
import argparse

# -----------------------------
# Path setup
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from ultralytics import YOLO
from phase3.code.rgbt_trainer import RGBTDetectionTrainer


# -----------------------------
# Main training
# -----------------------------
def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fusion_mode", type=str, default="vqc",
                         choices=["add", "concat", "vqc", "qinspired", "rgvqf"])
    parser.add_argument("--attention", type=str, default=None, choices=["none", "cbam", "cmaam"])
    parser.add_argument("--no_entanglement", action="store_true",
                         help="rgvqf only: disable ring-CNOT entanglement (no-entangle ablation)")
    parser.add_argument("--use_reliability", action="store_true",
                         help="rgvqf only: weight encoding by REM's per-modality reliability instead of fixed 0.5/0.5")
    parser.add_argument("--single_scale", action="store_true",
                         help="MQFP ablation: apply fusion_mode only at P5 (P3/P4 fall back to add)")
    parser.add_argument("--alpha_rel", type=float, default=0.0, help="composite loss weight on L_rel")
    parser.add_argument("--beta_q", type=float, default=0.0, help="composite loss weight on L_q")
    args = parser.parse_args()

    fusion_mode = args.fusion_mode
    print(f"[INFO] Fusion mode set to: {fusion_mode}")
    if args.attention is not None:
        print(f"[INFO] Attention mode set to: {args.attention}")
    if fusion_mode == "rgvqf" and args.no_entanglement:
        print("[INFO] RGVQF entanglement disabled (no-entangle ablation)")
    if fusion_mode == "rgvqf" and args.use_reliability:
        print("[INFO] RGVQF reliability weighting enabled (REM)")
    if args.single_scale:
        print("[INFO] Single-scale ablation: fusion_mode applied at P5 only (MQFP off)")
    if args.alpha_rel or args.beta_q:
        print(f"[INFO] Composite loss: alpha_rel={args.alpha_rel} beta_q={args.beta_q}")

    print("[INFO] Starting FINAL YOLOv11 + CBAM + RGB–Thermal Training")

    # Load YOLOv11 base
    yolo = YOLO("yolo11n.pt")

    ProposedRGBTTrainer = type(
        "ProposedRGBTTrainer",
        (RGBTDetectionTrainer,),
        {
            "fusion_mode": fusion_mode,
            "use_cbam": True,
            "attention": args.attention,
            "use_entanglement": not args.no_entanglement,
            "use_reliability": args.use_reliability,
            "multiscale": not args.single_scale,
            "alpha_rel": args.alpha_rel,
            "beta_q": args.beta_q,
        },
    )

    # Train
    yolo.train(
        trainer=ProposedRGBTTrainer,
        data="phase3/dataset_aligned/data.yaml",

        epochs=120,
        imgsz=960,
        batch=4,

        box=10.0,
        cls=0.3,
        dfl=1.5,

        # Keep paired RGB/thermal images spatially aligned.
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

        patience=40,
        device="mps",
        workers=2,

        name="yolov11L_rgb_ir_cbam_highAP",

        resume=False
    )




if __name__ == "__main__":
    main()
