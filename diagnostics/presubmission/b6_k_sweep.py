#!/usr/bin/env python3
"""
B6 refinement: choose the encoder gain k in x = pi*tanh(z/k), z = BN(Linear(p)), WITHOUT retraining.
On b6_pilot/runs/b6_on/weights/last.pt (trained with k=1), capture z per scale for
  - 100 val images, eval mode, batch 1 (a fixed random subset of val, seed 0 -- not 100 consecutive frames)
  - 100 TRAIN images, whole model in train mode, shuffled batches of 4 (the real train loader: seeded shuffle + train aug)
and for k in K_GRID report: eval saturation (|x|>0.9pi), train-mode saturation, eval std of x across images.
Choice rule: smallest k with eval saturation < 10% at every scale. CPU, read-only, nothing saved but the json/md.
    .venv/bin/python -u diagnostics/presubmission/b6_k_sweep.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

CKPT = os.path.join(HERE, "b6_pilot", "runs", "b6_on", "weights", "last.pt")
K_GRID = [1, 1.5, 2, 2.5, 3, 4]
FLAGS = dict(attn_channel_fix=True, aux_loss_batch_scale=True, rem_fused_levels_only=True, seeded_data_order=True,
             rgvqf_learned_encoder=True)
SET = dict(data=os.path.join(HERE, "data", "fusion", "data.yaml"), imgsz=960, batch=4, device="cpu", workers=0, rect=True,
           model="yolo11n.pt", plots=False, exist_ok=True, epochs=5, seed=0, deterministic=True, box=10.0, cls=0.3, dfl=1.5,
           mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
           perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
           project=os.path.join(HERE, "_k_scratch"), name="k")


def setup(mode):
    tr = make_trainer({**ABLATIONS["cbam_rgvqf"], **FLAGS})(overrides=SET)
    m = YOLO(CKPT).model.float()
    tr.model = m
    tr.set_model_attributes()
    zs = {}
    for lvl in ("p3", "p4", "p5"):
        getattr(m.fusion_backbone, f"fuse_{lvl}").encoder_bn.register_forward_hook(
            lambda _m, _i, o, l=lvl: zs.setdefault(l, []).append(o.detach().float().cpu()))
    m.train() if mode == "train" else m.eval()
    return tr, m, zs


def capture_eval():
    tr, m, zs = setup("eval")
    ds = tr.build_dataset(tr.data["val"], mode="val", batch=1)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:100].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, collate_fn=ds.collate_fn)
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    with torch.no_grad():
        for b in dl:
            m(v.preprocess(b)["img"])
    return {k: torch.cat(v_) for k, v_ in zs.items()}


def capture_train():
    tr, m, zs = setup("train")
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")  # shuffled (B5 seed 0), train aug
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= 25:
                break
            m(tr.preprocess_batch(b)["img"])
    return {k: torch.cat(v_) for k, v_ in zs.items()}


def main():
    ze, zt = capture_eval(), capture_train()
    rows = {}
    for k in K_GRID:
        for lvl in ("p3", "p4", "p5"):
            xe, xt = math.pi * torch.tanh(ze[lvl] / k), math.pi * torch.tanh(zt[lvl] / k)
            rows.setdefault(str(k), {})[lvl] = dict(
                eval_sat=float((xe.abs() > 0.9 * math.pi).float().mean()),
                train_sat=float((xt.abs() > 0.9 * math.pi).float().mean()),
                eval_x_std=float(xe.std(0).mean()), train_x_std=float(xt.std(0).mean()))
    ok = [k for k in K_GRID if all(rows[str(k)][l]["eval_sat"] < 0.10 for l in ("p3", "p4", "p5"))]
    chosen = min(ok) if ok else None
    z_stats = {lvl: dict(eval_z_std=float(ze[lvl].std(0).mean()), eval_z_absmean=float(ze[lvl].abs().mean()),
                         train_z_std=float(zt[lvl].std(0).mean()), train_z_absmean=float(zt[lvl].abs().mean()))
               for lvl in ("p3", "p4", "p5")}
    json.dump({"rows": rows, "chosen_k": chosen, "z_stats": z_stats, "n_eval": int(ze["p3"].shape[0]),
               "n_train": int(zt["p3"].shape[0])}, open(os.path.join(HERE, "b6_k_sweep.json"), "w"), indent=1)
    L = ["| k | scale | eval sat (|x|>0.9π) | train-mode sat | eval std of x | train std of x |\n|---|---|---|---|---|---|\n"]
    for k in K_GRID:
        for lvl in ("p3", "p4", "p5"):
            r = rows[str(k)][lvl]
            L.append(f"| {k} | {lvl} | {r['eval_sat']:.3f} | {r['train_sat']:.3f} | {r['eval_x_std']:.3f} | {r['train_x_std']:.3f} |\n")
    L.append(f"\nz statistics (k-independent): {json.dumps({l: {kk: round(vv, 3) for kk, vv in s.items()} for l, s in z_stats.items()})}\n")
    L.append(f"\n**Chosen k (smallest with eval saturation < 10% at every scale): {chosen}**\n")
    open(os.path.join(HERE, "b6_k_sweep.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
