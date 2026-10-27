#!/usr/bin/env python3
"""
Pre-flight Part A: read-only checks on EXISTING trained checkpoints.
No repo code changes; checkpoints only read; scratch writes under preflight/_scratch.

    .venv/bin/python -u diagnostics/presubmission/preflight/partA_checks.py --check a1
    ... --check a2 [--group grid|seeds]
    ... --check a3 | a4 | a6
    ... --check a5 --run abl_complete
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.model.attention.cmaam import CMAAM  # noqa: E402
from phase3.code.model.cbam import CBAM  # noqa: E402
from phase3.code.model.fusion.rgvqf import RGVQFFusion  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "presubmission", "preflight")
GRID = os.path.join(ROOT, "diagnostics", "step5_retrain", "runs")
SEEDS = os.path.join(ROOT, "diagnostics", "pilot_seeds", "runs")
NAME2KEY = {c["name"]: k for k, c in ABLATIONS.items()}
RGVQF_GRID = [c["name"] for c in ABLATIONS.values() if c["fusion_mode"] == "rgvqf"]
GRID_OV = dict(data="phase3/dataset_aligned/data.yaml", epochs=50, imgsz=960, batch=4, workers=0, rect=True,
               fraction=1.0, patience=20, pretrained=True, box=10.0, cls=0.3, dfl=1.5, mosaic=0.0, mixup=0.0,
               copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0,
               fliplr=0.0, close_mosaic=0, val=True, plots=False, model="yolo11n.pt", deterministic=True)


def ckpt_path(name, seed, which):
    base = os.path.join(GRID, name) if seed == 0 else os.path.join(SEEDS, f"seed{seed}", name)
    return os.path.join(base, "weights", which)


def make(name, seed=0, device="cpu", tag="x"):
    """Replays run_step5_retrain.train_config's construction order:
    YOLO('yolo11n.pt') -> trainer.__init__ (init_seeds(seed)) -> get_model(cfg, weights)."""
    base = YOLO("yolo11n.pt")
    tr = make_trainer(ABLATIONS[NAME2KEY[name]])(overrides={
        **GRID_OV, "seed": seed, "device": device, "project": os.path.join(OUT, "_scratch"),
        "name": f"{tag}_{name}_s{seed}", "exist_ok": True})
    init = tr.get_model(cfg=base.model.yaml, weights=base.model, verbose=False)
    return tr, init


def load_trained(path, tr, device, grad=False):
    m = YOLO(path).model.float().to(device)
    for n, p in m.named_parameters():
        p.requires_grad_(grad and ".dfl" not in n)
    tr.model = m
    tr.set_model_attributes()
    return m


def val_batches(tr, n_batches=None):
    dl = tr.get_dataloader(tr.data["val"], batch_size=4, rank=-1, mode="val")
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    for i, b in enumerate(dl):
        if n_batches is not None and i >= n_batches:
            break
        yield v.preprocess(b)


def rgvqf_runs():
    runs = [(n, 0) for n in RGVQF_GRID] + [("abl_cbam_rgvqf_proposed", s) for s in range(1, 8)]
    return runs


# ------------------------------------------------------------------ A1
def a1():
    rows = []
    for name, seed in rgvqf_runs():
        tr, init = make(name, seed, tag="a1")
        _, init_wrong = make(name, seed + 100, tag="a1w")  # wrong-seed control for the replay check
        for which in ("best.pt", "last.pt"):
            m = YOLO(ckpt_path(name, seed, which)).model.float()
            fb, fi, fw = m.fusion_backbone, init.fusion_backbone, init_wrong.fusion_backbone
            for lvl in ("p3", "p4", "p5"):
                mt, m0, mw = getattr(fb, f"fuse_{lvl}"), getattr(fi, f"fuse_{lvl}"), getattr(fw, f"fuse_{lvl}")
                if not isinstance(mt, RGVQFFusion):
                    continue
                qt, q0, qw = mt.q_weights.detach().flatten(), m0.q_weights.detach().flatten(), mw.q_weights.detach().flatten()
                d = qt - q0
                cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))  # noqa: E731
                rows.append(dict(run=name, seed=seed, ckpt=which, level=lvl,
                                 q_init_norm=round(float(q0.norm()), 5), q_trained_norm=round(float(qt.norm()), 5),
                                 q_L2_change=round(float(d.norm()), 5), q_maxabs_change=round(float(d.abs().max()), 5),
                                 rel_change=round(float(d.norm() / q0.norm()), 3),
                                 cos_init_trained=round(cos(q0, qt), 4), cos_wrongseed_trained=round(cos(qw, qt), 4),
                                 to_gate_W_norm=round(float(mt.to_gate.weight.norm()), 4),
                                 to_gate_init_norm=round(float(m0.to_gate.weight.norm()), 4)))
                print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(OUT, "a1_qweights.json"), "w"), indent=1)


# ------------------------------------------------------------------ A2
def a2(group, device):
    runs = [r for r in rgvqf_runs() if (r[1] == 0) == (group == "grid")]
    rows = []
    for name, seed in runs:
        tr, _ = make(name, seed, device=device, tag="a2")
        for which in ("best.pt", "last.pt"):
            m = load_trained(ckpt_path(name, seed, which), tr, device).eval()
            fuses = {lvl: getattr(m.fusion_backbone, f"fuse_{lvl}") for lvl in ("p3", "p4", "p5")}
            qs, gs = {k: [] for k in fuses}, {k: [] for k in fuses}
            with torch.no_grad():
                for b in val_batches(tr, n_batches=25):  # 100 images
                    m(b["img"])
                    for lvl, f in fuses.items():
                        if isinstance(f, RGVQFFusion):
                            q = f.last_q_out.float().cpu()
                            qs[lvl].append(q)
                            gs[lvl].append(torch.sigmoid(f.to_gate.cpu().float()(q)))
                            f.to_gate.to(device)
            for lvl in fuses:
                if not qs[lvl]:
                    continue
                q = torch.cat(qs[lvl])
                g = torch.cat(gs[lvl])
                rows.append(dict(run=name, seed=seed, ckpt=which, level=lvl, n_images=q.shape[0],
                                 qubit_std=[round(float(x), 4) for x in q.std(0)],
                                 qubit_mean=[round(float(x), 3) for x in q.mean(0)],
                                 mean_qubit_std=round(float(q.std(0).mean()), 4),
                                 gate_mean=round(float(g.mean()), 4),
                                 gate_std_across_images_mean_over_channels=round(float(g.std(0).mean()), 5),
                                 gate_std_across_images_max_over_channels=round(float(g.std(0).max()), 5)))
                print(rows[-1], flush=True)
    path = os.path.join(OUT, f"a2_qout_{group}.json")
    json.dump(rows, open(path, "w"), indent=1)


# ------------------------------------------------------------------ A3
def a3(n=50):
    name = "abl_complete"
    tr, _ = make(name, 0, tag="a3")
    m = load_trained(ckpt_path(name, 0, "last.pt"), tr, "cpu", grad=True)
    m.train()
    m.criterion = m.init_criterion()
    cap = {}
    orig = m.criterion.get_assigned_targets_and_loss

    def wrapped(preds, batch):
        out = orig(preds, batch)
        cap["fg"] = out[0][0].detach()
        return out
    m.criterion.get_assigned_targets_and_loss = wrapped
    n3, n4 = 96 * 120, 48 * 60
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
    rows = []
    for i, raw in enumerate(dl):
        if i >= n:
            break
        b = tr.preprocess_batch(raw)
        m.zero_grad(set_to_none=True)
        loss, _ = m(b)
        loss.sum().backward()
        fg = cap["fg"]
        p3, p4, p5 = int(fg[:, :n3].sum()), int(fg[:, n3:n3 + n4].sum()), int(fg[:, n3 + n4:].sum())
        cv2 = [p for nme, p in m.named_parameters() if nme.startswith("model.23.cv2.0.")]
        cv3 = [p for nme, p in m.named_parameters() if nme.startswith("model.23.cv3.0.")]
        zero = lambda ps: all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in ps)  # noqa: E731
        hb = b["bboxes"][:, 3] * b["img"][0].shape[2]  # box height in pixels (normalised h * H)
        rows.append(dict(batch=i, n_gt=int(b["cls"].shape[0]), p3_pos=p3, p4_pos=p4, p5_pos=p5,
                         p3_box_grad_zero=zero(cv2), p3_cls_grad_zero=zero(cv3),
                         min_gt_h_px=round(float(hb.min()), 1) if len(hb) else None))
        print(rows[-1], flush=True)
    zb = [r for r in rows if r["p3_box_grad_zero"]]
    summ = dict(batches=len(rows), p3_box_zero_grad_batches=len(zb),
                zero_grad_batches_all_have_p3_pos_0=all(r["p3_pos"] == 0 for r in zb),
                nonzero_grad_batches_all_have_p3_pos_gt0=all(r["p3_pos"] > 0 for r in rows if not r["p3_box_grad_zero"]),
                batches_with_p3_pos=sum(r["p3_pos"] > 0 for r in rows),
                mean_pos_per_batch={k: round(sum(r[k] for r in rows) / len(rows), 2) for k in ("p3_pos", "p4_pos", "p5_pos")},
                p3_cls_grad_zero_batches=sum(r["p3_cls_grad_zero"] for r in rows))
    print("SUMMARY", summ)
    json.dump({"rows": rows, "summary": summ}, open(os.path.join(OUT, "a3_p3_head.json"), "w"), indent=1)


# ------------------------------------------------------------------ A4
def a4():
    res = {}
    for seed in (0, 1):
        tr, model = make("abl_cbam_no_vqc", seed, tag="a4")
        tr.model = model
        tr.set_model_attributes()
        dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
        files, first = [], None
        for i, raw in enumerate(dl):
            if i >= 5:
                break
            files.append([os.path.basename(f) for f in raw["im_file"]])
            if i == 0:
                first = tr.preprocess_batch(raw)
        model.train()
        with torch.no_grad():
            loss, items = model(first)
        res[seed] = dict(files=files, loss=[round(float(x), 5) for x in loss], loss_items=[round(float(x), 5) for x in items],
                         rgb_sum=float(first["img"][0].sum()), cls_head_w_norm=float(model.model[23].cv3[0][2].weight.norm()),
                         cbam_W1_norm=float(model.fusion_backbone.rgb_cbam_p3.mlp[0].weight.norm()),
                         generator_seed_note="ultralytics/data/build.py:324-325 seeds DataLoader generator with a constant")
        print(seed, json.dumps(res[seed]), flush=True)
    res["same_data_order_first5"] = res[0]["files"] == res[1]["files"]
    res["same_batch1_pixels"] = res[0]["rgb_sum"] == res[1]["rgb_sum"]
    res["same_initial_loss"] = res[0]["loss"] == res[1]["loss"]
    res["same_cls_head_init"] = res[0]["cls_head_w_norm"] == res[1]["cls_head_w_norm"]
    print({k: v for k, v in res.items() if isinstance(k, str) and k.startswith("same")})
    json.dump(res, open(os.path.join(OUT, "a4_seeds.json"), "w"), indent=1)


# ------------------------------------------------------------------ A5
def a5(name, device):
    tr, _ = make(name, 0, device=device, tag="a5")
    m = load_trained(ckpt_path(name, 0, "last.pt"), tr, device).eval()
    fb = m.fusion_backbone
    stats = {}

    def hook(key):
        def h(_m, _i, o):
            o = o.detach().flatten(1).float().cpu()  # [B, hidden]
            s = stats.setdefault(key, {"active": 0, "total": 0, "ever": torch.zeros(o.shape[1], dtype=torch.bool)})
            s["active"] += int((o > 0).sum())
            s["total"] += o.numel()
            s["ever"] |= (o > 0).any(0)
        return h
    for nme, mod in fb.named_modules():
        if isinstance(mod, CBAM):
            mod.mlp[1].register_forward_hook(hook(f"{nme}.mlp"))
        elif isinstance(mod, CMAAM):
            mod.channel_mlp_rgb[1].register_forward_hook(hook(f"{nme}.rgb"))
            mod.channel_mlp_ir[1].register_forward_hook(hook(f"{nme}.ir"))
    t0 = time.time()
    with torch.no_grad():
        for b in val_batches(tr):
            m(b["img"])
    rows = {k: dict(active_fraction=round(v["active"] / v["total"], 5), units_ever_active=int(v["ever"].sum()),
                    hidden_units=int(v["ever"].numel()), activations_seen=v["total"]) for k, v in stats.items()}
    for k, v in rows.items():
        print(f"{name} {k:24s} {v}")
    print(f"[{time.time() - t0:.0f}s over full val]")
    json.dump(rows, open(os.path.join(OUT, f"a5_relu_{name}.json"), "w"), indent=1)


# ------------------------------------------------------------------ A6
def a6():
    name = "abl_complete"
    tr, init = make(name, 0, tag="a6")
    tr.model = init
    tr.set_model_attributes()
    init.train()
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
    b = tr.preprocess_batch(next(iter(dl)))
    loss, _ = init(b)
    loss.sum().backward()
    out = {"init_grads": {}, "trained": {}}
    for lvl in ("p3", "p4", "p5"):
        c = getattr(init.fusion_backbone, f"cmaam_{lvl}")
        for br in ("rgb", "ir"):
            mlp = getattr(c, f"channel_mlp_{br}")
            out["init_grads"][f"{lvl}.{br}"] = dict(W1_grad_norm=float(mlp[0].weight.grad.norm()),
                                                     W2_grad_norm=float(mlp[2].weight.grad.norm()),
                                                     W2_init_norm=float(mlp[2].weight.norm()))
    print("INIT GRADS", json.dumps(out["init_grads"], indent=1))
    # trained: descriptor sign, pre-activation sign with trained W1 vs re-created init W1 on the SAME inputs
    m = load_trained(ckpt_path(name, 0, "last.pt"), tr, "cpu").eval()
    caps = {}

    def pre(key):
        def h(_m, inp):
            caps.setdefault(key, []).append(inp[0].detach().flatten(1))
        return h
    for lvl in ("p3", "p4", "p5"):
        c = getattr(m.fusion_backbone, f"cmaam_{lvl}")
        for br in ("rgb", "ir"):
            getattr(c, f"channel_mlp_{br}").register_forward_pre_hook(pre(f"{lvl}.{br}"))
    with torch.no_grad():
        for bb in val_batches(tr, n_batches=25):
            m(bb["img"])
    for lvl in ("p3", "p4", "p5"):
        for br in ("rgb", "ir"):
            key = f"{lvl}.{br}"
            d = torch.cat(caps[key])  # [N, 4C]
            W1t = getattr(getattr(m.fusion_backbone, f"cmaam_{lvl}"), f"channel_mlp_{br}")[0].weight.detach().flatten(1)
            W10 = getattr(getattr(init.fusion_backbone, f"cmaam_{lvl}"), f"channel_mlp_{br}")[0].weight.detach().flatten(1)
            zt, z0 = d @ W1t.t(), d @ W10.t()
            out["trained"][key] = dict(
                descriptor_frac_positive=round(float((d > 0).float().mean()), 4),
                preact_trained_frac_positive=round(float((zt > 0).float().mean()), 4),
                preact_trained_max=round(float(zt.max()), 4),
                preact_initW1_frac_positive=round(float((z0 > 0).float().mean()), 4),
                W1_init_norm=round(float(W10.norm()), 4), W1_trained_norm=round(float(W1t.norm()), 4),
                cos_W1_init_trained=round(float(torch.nn.functional.cosine_similarity(W10.flatten(), W1t.flatten(), 0)), 4),
                W1_rowsum_mean_init=round(float(W10.sum(1).mean()), 4), W1_rowsum_mean_trained=round(float(W1t.sum(1).mean()), 4))
            print(key, out["trained"][key], flush=True)
    json.dump(out, open(os.path.join(OUT, "a6_relu_death.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["a1", "a2", "a3", "a4", "a5", "a6"])
    ap.add_argument("--group", default="grid", choices=["grid", "seeds"])
    ap.add_argument("--run", default="abl_complete")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"a1": a1, "a2": lambda: a2(a.group, a.device), "a3": a3, "a4": a4,
     "a5": lambda: a5(a.run, a.device), "a6": a6}[a.check]()
