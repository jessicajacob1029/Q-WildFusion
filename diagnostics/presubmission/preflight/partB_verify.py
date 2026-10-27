#!/usr/bin/env python3
"""
Pre-flight Part B verification (fix flags ON).

  --check gates    : B1 step-0 channel gates (must be 1.0 within 1e-3), all levels, cbam_add + cmaam_add
  --check train    : B1 short run (fraction 0.05, 1 epoch, val off, MPS) then one fwd/bwd on the final
                     (non-EMA) model: channel-MLP pre-activation >0 fraction and W1/W2 grads at every level
                     --config cbam_add | cmaam_add
  --check recover  : trained DEAD abl_complete last.pt, ReLU swapped for LeakyReLU(0.1): do W1 grads return?
  --check b2       : loss vector, complete config, aux_loss_batch_scale off vs on, same batch
  --check b3b4     : single-scale REM instantiation off vs on + param_report
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.model.attention.cmaam import CMAAM  # noqa: E402
from phase3.code.model.cbam import CBAM  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "presubmission", "preflight")
OV = dict(data="phase3/dataset_aligned/data.yaml", imgsz=960, batch=4, workers=0, rect=True, seed=0,
          deterministic=True, model="yolo11n.pt", plots=False, epochs=1, box=10.0, cls=0.3, dfl=1.5,
          mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
          perspective=0.0, flipud=0.0, fliplr=0.0, close_mosaic=0, pretrained=True, exist_ok=True)


def trainer_cls(key, **flags):
    return type("PreflightTrainer", (make_trainer(ABLATIONS[key]),), dict(flags))


def build(key, device="cpu", tag="v", **flags):
    base = YOLO("yolo11n.pt")
    tr = trainer_cls(key, **flags)(overrides={**OV, "device": device, "project": os.path.join(OUT, "_scratch"),
                                              "name": f"{tag}_{key}"})
    m = tr.get_model(cfg=base.model.yaml, weights=base.model, verbose=False)
    tr.model = m
    tr.set_model_attributes()
    return tr, m


def channel_gates(fb, inputs):
    """Recompute each channel gate from the captured module inputs."""
    out = {}
    with torch.no_grad():
        for name, mod in fb.named_modules():
            if isinstance(mod, CBAM) and name in inputs:
                x = inputs[name][0]
                out[name] = 2 * torch.sigmoid(mod.mlp(mod.avgPool(x)) + mod.mlp(mod.maxPool(x)))
            elif isinstance(mod, CMAAM) and name in inputs:
                r, t = inputs[name]
                j = torch.cat([mod.avg_pool(r), mod.max_pool(r), mod.avg_pool(t), mod.max_pool(t)], 1)
                out[name + ".rgb"] = 2 * torch.sigmoid(mod.channel_mlp_rgb(j))
                out[name + ".ir"] = 2 * torch.sigmoid(mod.channel_mlp_ir(j))
    return out


def capture_inputs(fb):
    caps = {}
    for name, mod in fb.named_modules():
        if isinstance(mod, (CBAM, CMAAM)):
            mod.register_forward_pre_hook(lambda _m, inp, n=name: caps.__setitem__(n, [t.detach() for t in inp]))
    return caps


def channel_mlps(fb):
    for name, mod in fb.named_modules():
        if isinstance(mod, CBAM):
            yield name, mod.mlp
        elif isinstance(mod, CMAAM):
            yield name + ".rgb", mod.channel_mlp_rgb
            yield name + ".ir", mod.channel_mlp_ir


def liveness_and_grads(model, batch):
    """One fwd/bwd; per channel MLP: fraction of hidden pre-activations > 0, W1/W2 grad norms."""
    fb = model.fusion_backbone
    pre = {}
    hooks = [mlp[0].register_forward_hook(lambda _m, _i, o, n=n: pre.__setitem__(n, o.detach()))
             for n, mlp in channel_mlps(fb)]
    model.zero_grad(set_to_none=True)
    model.train()
    loss, _ = model(batch)
    loss.sum().backward()
    for h in hooks:
        h.remove()
    rows = {}
    for n, mlp in channel_mlps(fb):
        g1, g2 = mlp[0].weight.grad, mlp[2].weight.grad
        rows[n] = dict(act=type(mlp[1]).__name__, preact_pos_frac=round(float((pre[n] > 0).float().mean()), 4),
                       W1_grad=None if g1 is None else float(g1.norm()), W2_grad=None if g2 is None else float(g2.norm()))
    return rows


def check_gates():
    res = {}
    for key in ("cbam_add", "cmaam_add"):
        for fix in (False, True):
            tr, m = build(key, tag=f"gates{int(fix)}", attn_channel_fix=fix)
            caps = capture_inputs(m.fusion_backbone)
            dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
            b = tr.preprocess_batch(next(iter(dl)))
            m.eval()
            with torch.no_grad():
                m(b["img"])
            g = channel_gates(m.fusion_backbone, caps)
            dev = {k: float((v - 1).abs().max()) for k, v in g.items()}
            res[f"{key}_fix{fix}"] = dev
            print(f"{key} fix={fix}: max|gate-1| per level: " + ", ".join(f"{k}={v:.2e}" for k, v in dev.items())
                  + f"  -> overall {max(dev.values()):.2e} {'PASS' if max(dev.values()) <= 1e-3 else 'FAIL'} (<=1e-3)")
    json.dump(res, open(os.path.join(OUT, "partB_gates.json"), "w"), indent=1)


def check_train(key, device):
    steps = {"n": 0}

    class Counting(trainer_cls(key, attn_channel_fix=True)):
        def optimizer_step(self):
            steps["n"] += 1
            super().optimizer_step()

    t0 = time.time()
    y = YOLO("yolo11n.pt")
    y.train(trainer=Counting, **{**OV, "device": device, "fraction": 0.05, "val": False, "patience": 50,
                                 "project": os.path.join(OUT, "_scratch"), "name": f"b1train_{key}"})
    tr = y.trainer
    print(f"[train] {key}: {time.time() - t0:.0f}s, optimizer steps={steps['n']}, "
          f"train batches={len(tr.train_loader)}")
    b = tr.preprocess_batch(next(iter(tr.train_loader)))
    rows = liveness_and_grads(tr.model, b)
    ok = True
    for n, r in rows.items():
        good = r["preact_pos_frac"] > 0 and (r["W1_grad"] or 0) > 0 and (r["W2_grad"] or 0) > 0
        ok &= good
        print(f"  {n:22s} {r}  {'OK' if good else 'FAIL'}")
    print(f"[train] {key} channel MLPs alive with non-zero W1/W2 grads at every level: {'PASS' if ok else 'FAIL'}")
    json.dump({"steps": steps["n"], "rows": rows, "pass": ok}, open(os.path.join(OUT, f"partB_train_{key}.json"), "w"), indent=1)


def check_recover():
    tr, _ = build("complete", tag="recover")
    m = YOLO("diagnostics/step5_retrain/runs/abl_complete/weights/last.pt").model.float()
    for n, p in m.named_parameters():
        p.requires_grad_(".dfl" not in n)
    tr.model = m
    tr.set_model_attributes()
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
    b = tr.preprocess_batch(next(iter(dl)))
    before = liveness_and_grads(m, b)
    for _, mlp in channel_mlps(m.fusion_backbone):
        mlp[1] = nn.LeakyReLU(0.1, inplace=True)
    after = liveness_and_grads(m, b)
    for n in before:
        print(f"  {n:16s} ReLU: W1_grad={before[n]['W1_grad']:.2e} W2_grad={before[n]['W2_grad']:.2e} | "
              f"LeakyReLU: W1_grad={after[n]['W1_grad']:.2e} W2_grad={after[n]['W2_grad']:.2e}")
    json.dump({"relu": before, "leaky": after}, open(os.path.join(OUT, "partB_recover.json"), "w"), indent=1)


def check_b2():
    out = {}
    for flag in (False, True):
        torch.manual_seed(0)
        tr, m = build("complete", tag=f"b2_{int(flag)}", aux_loss_batch_scale=flag)
        dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
        b = tr.preprocess_batch(next(iter(dl)))
        m.train()
        loss, items = m(b)
        out[flag] = [float(x) for x in loss]
        print(f"aux_loss_batch_scale={flag}: loss vector [box*B, cls*B, dfl*B, a*L_rel, b*L_q] = "
              f"{[round(x, 5) for x in out[flag]]}  sum={sum(out[flag]):.5f}  logged items={[round(float(x), 5) for x in items]}")
    ratio = [out[True][i] / out[False][i] for i in (3, 4)]
    print(f"aux terms ON/OFF ratio = {[round(r, 4) for r in ratio]} (expected B=4); det terms identical: "
          f"{out[True][:3] == out[False][:3]}")
    json.dump(out, open(os.path.join(OUT, "partB_b2.json"), "w"), indent=1)


def check_b3b4():
    for flag in (False, True):
        tr, m = build("cmaam_rgvqf_rem_single_scale", tag=f"b3_{int(flag)}", rem_fused_levels_only=flag)
        rems = [n for n in ("rem_p3", "rem_p4", "rem_p5") if hasattr(m.fusion_backbone, n)]
        print(f"single-scale rem_fused_levels_only={flag}: REM modules={rems} param_report={m.param_report()}")
    for key in ABLATIONS:
        _, m = build(key, tag="b4")
        print(f"B4 {ABLATIONS[key]['name']:28s} {m.param_report()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["gates", "train", "recover", "b2", "b3b4"])
    ap.add_argument("--config", default="cmaam_add")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    {"gates": check_gates, "train": lambda: check_train(a.config, a.device), "recover": check_recover,
     "b2": check_b2, "b3b4": check_b3b4}[a.check]()
