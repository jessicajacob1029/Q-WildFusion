#!/usr/bin/env python3
"""
B6 pilot addendum (read-only on the pilot's last.pt files, CPU):
  0. proof that aug hyperparameters don't reach the val pipeline: val dataset transforms
     built with Ultralytics defaults (mosaic 1, fliplr .5, ...) vs grid settings
  1. A2 in eval mode, batch 1 AND batch 4 (eval is per-sample, so they must match)
  2. A2 with the whole model in train mode, batch 4 (what the circuit sees while training;
     BN uses batch stats); fresh checkpoint load, nothing saved
  3. per scale: fraction of encoder angles |x| > 0.9*pi, encoder BatchNorm running_var
  4. per-epoch mAP50-95 for both runs (results.csv)
Appends to b6_pilot/summary.md and writes b6_pilot/addendum.json.

    .venv/bin/python -u diagnostics/presubmission/b6_pilot_addendum.py
"""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.model.fusion.rgvqf import RGVQFFusion  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

PILOT = os.path.join(ROOT, "diagnostics", "presubmission", "b6_pilot")
GRID_AUG = dict(box=10.0, cls=0.3, dfl=1.5, mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0,
                scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0)
FLAGS = {"b6_off": dict(attn_channel_fix=True, aux_loss_batch_scale=True, rem_fused_levels_only=True,
                        seeded_data_order=True, rgvqf_learned_encoder=False)}
FLAGS["b6_on"] = dict(FLAGS["b6_off"], rgvqf_learned_encoder=True)


def make_tr(name, aug):
    cls = make_trainer({**ABLATIONS["cbam_rgvqf"], **FLAGS[name]})
    return cls(overrides=dict(data="phase3/dataset_aligned/data.yaml", imgsz=960, batch=4, device="cpu", workers=0,
                              rect=True, model="yolo11n.pt", plots=False, exist_ok=True, epochs=5,
                              project=os.path.join(PILOT, "_scratch"), name=f"addendum_{name}", **aug))


def load(name, tr):
    m = YOLO(os.path.join(PILOT, "runs", name, "weights", "last.pt")).model.float()
    tr.model = m
    tr.set_model_attributes()
    return m


def val_loader(tr, bs):
    dl = tr.get_dataloader(tr.data["val"], batch_size=bs, rank=-1, mode="val")
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    return dl, v


def measure(name, bs, model_mode):
    tr = make_tr(name, GRID_AUG)
    m = load(name, tr)
    m.train() if model_mode == "train" else m.eval()
    caps = {}
    for lvl in ("p3", "p4", "p5"):
        f = getattr(m.fusion_backbone, f"fuse_{lvl}")
        if not isinstance(f, RGVQFFusion):
            continue
        orig = f._descriptor

        def wrapped(*a, _o=orig, _l=lvl, **k):
            x = _o(*a, **k)
            caps.setdefault(_l, {"x": [], "q": [], "g": []})["x"].append(x.detach().float().cpu())
            return x
        f._descriptor = wrapped
    dl, v = val_loader(tr, bs)
    n = 0
    with torch.no_grad():
        for b in dl:
            if n >= 100:
                break
            m(v.preprocess(b)["img"])
            n += len(b["im_file"])
            for lvl in caps:
                f = getattr(m.fusion_backbone, f"fuse_{lvl}")
                q = f.last_q_out.float()
                caps[lvl]["q"].append(q.cpu())
                caps[lvl]["g"].append(torch.sigmoid(f.to_gate(q)).cpu())
    out = {}
    for lvl, c in caps.items():
        x, q, g = (torch.cat(c[k])[:100] for k in ("x", "q", "g"))
        f = getattr(m.fusion_backbone, f"fuse_{lvl}")
        bn = getattr(f, "encoder_bn", None)
        out[lvl] = dict(n_images=x.shape[0], mean_qubit_std=float(q.std(0).mean()),
                        gate_std_mean=float(g.std(0).mean()), gate_std_max=float(g.std(0).max()),
                        x_saturated_frac=float((x.abs() > 0.9 * torch.pi).float().mean()),
                        x_std_mean=float(x.std(0).mean()),
                        bn_running_var=None if bn is None else [round(float(t), 4) for t in bn.running_var],
                        bn_running_mean=None if bn is None else [round(float(t), 4) for t in bn.running_mean])
    return out


def main():
    res = {}
    # 0. val pipeline is independent of augmentation hyperparameters
    t_def = make_tr("b6_on", {})  # Ultralytics defaults: mosaic 1.0, fliplr 0.5, scale 0.5, translate 0.1, ...
    t_def.model = load("b6_on", t_def)
    t_grid = make_tr("b6_on", GRID_AUG)
    t_grid.model = load("b6_on", t_grid)
    tf = {k: repr(t.get_dataloader(t.data["val"], batch_size=4, rank=-1, mode="val").dataset.transforms)
          for k, t in (("defaults", t_def), ("grid", t_grid))}
    res["val_transforms_identical"] = tf["defaults"] == tf["grid"]
    res["val_transforms"] = tf["grid"]
    print("val transforms identical under default vs grid aug hyps:", res["val_transforms_identical"])
    print("  ", tf["grid"].replace("\n", " ")[:300])
    # 1-3
    for name in ("b6_off", "b6_on"):
        for label, bs, mode in (("eval_b1", 1, "eval"), ("eval_b4", 4, "eval"), ("train_b4", 4, "train")):
            res.setdefault(name, {})[label] = measure(name, bs, mode)
            print(name, label, json.dumps({k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                               if kk not in ("bn_running_mean",)} for k, v in res[name][label].items()}))
    # 4
    for name in ("b6_off", "b6_on"):
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(os.path.join(PILOT, "runs", name, "results.csv")))]
        res[name]["per_epoch"] = [(int(float(r["epoch"])), float(r["metrics/mAP50(B)"]), float(r["metrics/mAP50-95(B)"])) for r in rows]
    json.dump(res, open(os.path.join(PILOT, "addendum.json"), "w"), indent=1)

    L = ["\n## Addendum\n\n",
         f"- Both training runs used the grid settings (`runs/*/args.yaml` and `manifest.json`): box 10, cls 0.3, dfl 1.5, "
         "mosaic/mixup/copy_paste/degrees/translate/scale/shear/perspective/flipud/fliplr 0, close_mosaic 0, epochs 5, rect, imgsz 960. "
         "The default-valued trainer banners in run.log (`reval_*`, `a2`) are post-training evaluation helpers that build only a "
         f"val loader. Val transforms are identical under default vs grid aug hyperparameters: **{res['val_transforms_identical']}** "
         "(LetterBox + Format only).\n",
         "- The A2 numbers in the table above were measured in **eval mode with batch 4**. Re-measured below in eval mode with batch 1 "
         "(must equal batch 4, since eval mode has no batch coupling), and with the whole model in **train mode** (batch 4, BN batch stats).\n\n",
         "| run | scale | <Z> std eval b1 | eval b4 | train b4 | gate std (mean / max ch) eval b1 | train b4 | |x|>0.9π eval b1 | train b4 |\n",
         "|---|---|---|---|---|---|---|---|---|\n"]
    for name in ("b6_off", "b6_on"):
        for lvl in ("p3", "p4", "p5"):
            e1, e4, t4 = (res[name][k][lvl] for k in ("eval_b1", "eval_b4", "train_b4"))
            L.append(f"| {name} | {lvl} | {e1['mean_qubit_std']:.4f} | {e4['mean_qubit_std']:.4f} | {t4['mean_qubit_std']:.4f} | "
                     f"{e1['gate_std_mean']:.5f} / {e1['gate_std_max']:.5f} | {t4['gate_std_mean']:.5f} / {t4['gate_std_max']:.5f} | "
                     f"{e1['x_saturated_frac']:.4f} | {t4['x_saturated_frac']:.4f} |\n")
    L.append("\nEncoder BatchNorm running_var / running_mean (b6_on, per scale, 8 dims):\n\n")
    for lvl in ("p3", "p4", "p5"):
        e = res["b6_on"]["eval_b1"][lvl]
        L.append(f"- {lvl}: var {e['bn_running_var']} · mean {e['bn_running_mean']}\n")
    L.append("\nPer-epoch in-loop mAP50 / mAP50-95 (fixed validator, EMA weights):\n\n| epoch | b6_off mAP50 | b6_off mAP50-95 | b6_on mAP50 | b6_on mAP50-95 |\n|---|---|---|---|---|\n")
    for (e, a50, a5095), (_, b50, b5095) in zip(res["b6_off"]["per_epoch"], res["b6_on"]["per_epoch"]):
        L.append(f"| {e} | {a50:.4f} | {a5095:.4f} | {b50:.4f} | {b5095:.4f} |\n")
    open(os.path.join(PILOT, "summary.md"), "a").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
