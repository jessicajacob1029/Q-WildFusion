#!/usr/bin/env python3
"""
Checks for B5 (seeded data order), B6-diagnose (why x is image-independent),
B6-fix (learned encoder) and B7 (classical control). CPU; checkpoints read-only;
dataset read at fraction 1.0 only (no label-cache rewrite).

    .venv/bin/python -u diagnostics/presubmission/preflight/b5b6b7_checks.py --check b5|b6diag|b6fix|b7
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics", "presubmission"))
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import phase3.code.model.fusion.rgvqf as rgvqf_mod  # noqa: E402
from phase3.code.model.fusion.rgvqf import N_QUBITS, RGVQFFusion  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, EXTRA_ABLATIONS, make_trainer  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "presubmission", "preflight")
ALL = {**ABLATIONS, **EXTRA_ABLATIONS}
OV = dict(data="phase3/dataset_aligned/data.yaml", imgsz=960, batch=4, workers=0, rect=True, deterministic=True,
          model="yolo11n.pt", plots=False, epochs=1, box=10.0, cls=0.3, dfl=1.5, mosaic=0.0, mixup=0.0,
          copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0,
          fliplr=0.0, close_mosaic=0, pretrained=True, exist_ok=True, device="cpu", fraction=1.0)
NEW_FLAGS = dict(attn_channel_fix=True, aux_loss_batch_scale=True, rem_fused_levels_only=True, seeded_data_order=True)


def trainer(key, seed=0, tag="x", **flags):
    cls = type("T", (make_trainer({**ALL[key], **flags}),), {})
    return cls(overrides={**OV, "seed": seed, "project": os.path.join(OUT, "_scratch"), "name": f"{tag}_{key}_s{seed}"})


def init_model(tr):
    base = YOLO("yolo11n.pt")
    m = tr.get_model(cfg=base.model.yaml, weights=base.model, verbose=False)
    tr.model = m
    tr.set_model_attributes()
    return m


def first_files(tr, n=5):
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
    out = []
    for i, b in enumerate(dl):
        if i >= n:
            break
        out.append([os.path.basename(f) for f in b["im_file"]])
    return out


def val_batches(tr, n_batches=25):
    dl = tr.get_dataloader(tr.data["val"], batch_size=4, rank=-1, mode="val")
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    for i, b in enumerate(dl):
        if i >= n_batches:
            break
        yield v.preprocess(b)


# ------------------------------------------------------------------ B5
def check_b5():
    old_order = json.load(open(os.path.join(OUT, "a4_seeds.json")))["0"]["files"]
    res = {}
    for flag in (False, True):
        for seed in (0, 1):
            tr = trainer("cbam_add", seed, tag=f"b5_{int(flag)}", seeded_data_order=flag)
            init_model(tr)
            res[(flag, seed)] = first_files(tr)
    # baseline subclass (stock DetectionTrainer + pad0 + B5), RGB-only data at fraction 1.0
    from baseline_trainer import BaselineTrainer
    base_res = {}
    for flag in (False, True):
        for seed in (0, 1):
            cls = type("B", (BaselineTrainer,), {"seeded_data_order": flag})
            tr = cls(overrides={**OV, "data": "diagnostics/rgb_only.yaml", "seed": seed,
                                "project": os.path.join(OUT, "_scratch"), "name": f"b5base_{int(flag)}_s{seed}"})
            tr.model = tr.get_model(cfg=YOLO("yolo11n.pt").model.yaml, weights=YOLO("yolo11n.pt").model, verbose=False)
            base_res[(flag, seed)] = first_files(tr)
    summary = {
        "fusion: flag OFF seed0 == old order (a4)": res[(False, 0)] == old_order,
        "fusion: flag OFF seed0 == seed1 (old behaviour)": res[(False, 0)] == res[(False, 1)],
        "fusion: flag ON  seed0 != seed1": res[(True, 0)] != res[(True, 1)],
        "baseline: flag OFF seed0 == old order (a4)": base_res[(False, 0)] == old_order,
        "baseline: flag ON  seed0 != seed1": base_res[(True, 0)] != base_res[(True, 1)],
        "fusion ON seed s == baseline ON seed s (same order for paired comparisons)":
            res[(True, 0)] == base_res[(True, 0)] and res[(True, 1)] == base_res[(True, 1)],
    }
    for k, v in summary.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  fusion ON seed0 batch1:", res[(True, 0)][0], "| seed1 batch1:", res[(True, 1)][0])
    json.dump({"summary": summary, "fusion": {str(k): v for k, v in res.items()},
               "baseline": {str(k): v for k, v in base_res.items()}}, open(os.path.join(OUT, "b5.json"), "w"), indent=1)


# ------------------------------------------------------------------ B6 diagnose
def capture_descriptor_stages(model):
    store = {}

    def hook(name):
        def h(mod, args, kwargs):
            rgb, ir = args[0], args[1]
            w_rgb = kwargs.get("w_rgb", args[2] if len(args) > 2 else 0.5)
            w_th = kwargs.get("w_th", args[3] if len(args) > 3 else 0.5)
            with torch.no_grad():
                p = torch.cat([w_rgb * rgb, w_th * ir], dim=1).mean(dim=(2, 3)).float().cpu()  # [B, 2C]
                pre = F.adaptive_avg_pool1d(p.unsqueeze(1), N_QUBITS).squeeze(1)  # [B, 8]
                if getattr(mod, "learned_encoder", False):
                    pre = mod.encoder_bn(mod.encoder(p.to(rgb.device))).float().cpu()
                x = torch.pi * torch.tanh(pre)
            s = store.setdefault(name, {"p": [], "pre": [], "x": [], "q": []})
            s["p"].append(p)
            s["pre"].append(pre)
            s["x"].append(x)
        return h

    for lvl in ("p3", "p4", "p5"):
        f = getattr(model.fusion_backbone, f"fuse_{lvl}")
        if isinstance(f, RGVQFFusion):
            f.register_forward_pre_hook(hook(lvl), with_kwargs=True)
            f.register_forward_hook(lambda _m, _i, _o, n=lvl, mm=f: store[n]["q"].append(mm.last_q_out.detach().float().cpu()))
    return store


def summarize(store):
    rows = {}
    for lvl, s in store.items():
        p, pre, x, q = (torch.cat(s[k]) for k in ("p", "pre", "x", "q"))
        C2 = p.shape[1]
        g = C2 // N_QUBITS
        rows[lvl] = dict(
            n_images=p.shape[0],
            p_mean_abs=round(float(p.abs().mean()), 4),
            p_std_across_images_mean_over_dims=round(float(p.std(0).mean()), 4),
            p_cv=round(float((p.std(0) / p.abs().mean(0).clamp_min(1e-8)).mean()), 4),
            pre_std_per_dim=[round(float(v), 5) for v in pre.std(0)],
            pre_std_mean=round(float(pre.std(0).mean()), 5),
            x_std_per_dim=[round(float(v), 5) for v in x.std(0)],
            x_std_mean=round(float(x.std(0).mean()), 5),
            x_range=[round(float(x.min()), 3), round(float(x.max()), 3)],
            x_saturated_frac=round(float((x.abs() > 0.9 * torch.pi).float().mean()), 4),
            q_std_mean=round(float(q.std(0).mean()), 5),
            # hypothesis test: variance kept by averaging a group of channels vs. by one channel
            group_mean_std_over_mean_member_std=round(float(
                pre.std(0).mean() / p.reshape(p.shape[0], N_QUBITS, g).std(0).mean()), 4),
            # an untrained random Linear(2C,8) on the same p, for scale: how much variation is available
            random_linear_std_mean=round(float((p @ (torch.randn(C2, N_QUBITS, generator=torch.Generator().manual_seed(0))
                                                     / C2 ** 0.5)).std(0).mean()), 5),
        )
    return rows


def run_store(tr, model, n_batches=25):
    store = capture_descriptor_stages(model)
    model.eval()
    with torch.no_grad():
        for b in val_batches(tr, n_batches):
            model(b["img"])
    return summarize(store)


def check_b6diag():
    out = {}
    for key, path in (("cbam_rgvqf", "diagnostics/step5_retrain/runs/abl_cbam_rgvqf_proposed/weights/last.pt"),
                      ("complete", "diagnostics/step5_retrain/runs/abl_complete/weights/last.pt")):
        tr = trainer(key, tag="b6diag")
        m = YOLO(path).model.float()
        tr.model = m
        tr.set_model_attributes()
        out[key] = run_store(tr, m)
        for lvl, r in out[key].items():
            print(f"{key} {lvl}: p |mean|={r['p_mean_abs']} p std(img)={r['p_std_across_images_mean_over_dims']} "
                  f"(CV {r['p_cv']}) | pre-tanh std={r['pre_std_mean']} | x std={r['x_std_mean']} x∈{r['x_range']} | "
                  f"<Z> std={r['q_std_mean']} | group-mean std / member std = {r['group_mean_std_over_mean_member_std']} | "
                  f"random-Linear std={r['random_linear_std_mean']}")
    json.dump(out, open(os.path.join(OUT, "b6_diagnose.json"), "w"), indent=1)


# ------------------------------------------------------------------ B6 fix
def check_b6fix():
    out = {}
    for flag in (False, True):
        tr = trainer("cbam_rgvqf", tag=f"b6fix{int(flag)}", rgvqf_learned_encoder=flag, **NEW_FLAGS)
        m = init_model(tr)
        out[f"init_flag{flag}_eval"] = run_store(tr, m)
        m.train()  # train-mode BN uses batch stats: what the circuit sees during training
        store = capture_descriptor_stages(m)
        with torch.no_grad():
            for b in val_batches(tr, 25):
                m(b["img"])
        out[f"init_flag{flag}_trainmodeBN"] = summarize(store)
        for mode in ("eval", "trainmodeBN"):
            for lvl, r in out[f"init_flag{flag}_{mode}"].items():
                print(f"B6={flag} [{mode:11s}] {lvl}: <Z> std per qubit mean={r['q_std_mean']:.5f} | x std={r['x_std_mean']:.4f} "
                      f"x∈{r['x_range']} saturated={r['x_saturated_frac']}")
    # structural checks with the flag ON
    tr = trainer("cbam_rgvqf", tag="b6struct", rgvqf_learned_encoder=True, **NEW_FLAGS)
    m = init_model(tr)
    f = m.fusion_backbone.fuse_p3
    print("B6 ON modules:", [n for n, _ in f.named_children()], "| param_report:", m.param_report())
    m.train()
    r = torch.randn(1, 128, 8, 10)
    y = f(r, r.clone())  # batch of 1 in train mode (last LLVIP batch): must not raise
    print("B6 ON train-mode B=1 forward OK:", tuple(y.shape))
    with torch.no_grad():  # make the zero-init gate non-zero so encoder gradients are observable
        for lvl in ("p3", "p4", "p5"):
            getattr(m.fusion_backbone, f"fuse_{lvl}").to_gate.weight.normal_(0, 0.1)
    dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
    b = tr.preprocess_batch(next(iter(dl)))
    loss, _ = m(b)
    loss.sum().backward()
    for lvl in ("p3", "p4", "p5"):
        ff = getattr(m.fusion_backbone, f"fuse_{lvl}")
        print(f"  {lvl} grad norms: encoder.W={float(ff.encoder.weight.grad.norm()):.3e} "
              f"encoder_bn.weight={float(ff.encoder_bn.weight.grad.norm()):.3e} q_weights={float(ff.q_weights.grad.norm()):.3e}")
    json.dump(out, open(os.path.join(OUT, "b6_fix.json"), "w"), indent=1)


# ------------------------------------------------------------------ B7
def check_b7():
    calls = {"n": 0}
    orig = rgvqf_mod._rgvqf_circuit

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    rgvqf_mod._rgvqf_circuit = counted
    tr_q = trainer("cbam_rgvqf", tag="b7q", rgvqf_learned_encoder=True, **NEW_FLAGS)
    mq = init_model(tr_q)
    tr_c = trainer("cbam_rgvqf_classical_ctrl", tag="b7c", **NEW_FLAGS)
    mc = init_model(tr_c)
    sq, sc = mq.state_dict(), mc.state_dict()
    shared = [k for k in sc if k in sq]
    only_q = [k for k in sq if k not in sc]
    only_c = [k for k in sc if k not in sq]
    same = all(torch.equal(sq[k], sc[k]) for k in shared)
    print(f"B7 vs cbam_rgvqf(B6 ON), same seed: shared tensors={len(shared)} all identical={same}; "
          f"only in quantum={only_q}; only in control={only_c}")
    print("B7 param_report:", mc.param_report(), "| quantum variant:", mq.param_report())
    dl = tr_c.get_dataloader(tr_c.data["train"], batch_size=4, rank=-1, mode="train")
    b = tr_c.preprocess_batch(next(iter(dl)))
    with torch.no_grad():
        for lvl in ("p3", "p4", "p5"):
            getattr(mc.fusion_backbone, f"fuse_{lvl}").to_gate.weight.normal_(0, 0.1)
    calls["n"] = 0
    mc.train()
    loss, _ = mc(b)
    loss.sum().backward()
    print(f"B7 forward: QNode calls={calls['n']} loss finite={bool(torch.isfinite(loss).all())} loss={[round(float(x), 4) for x in loss]}")
    for lvl in ("p3", "p4", "p5"):
        ff = getattr(mc.fusion_backbone, f"fuse_{lvl}")
        print(f"  {lvl}: classical_q.W grad={float(ff.classical_q.weight.grad.norm()):.3e} "
              f"encoder.W grad={float(ff.encoder.weight.grad.norm()):.3e} has q_weights={hasattr(ff, 'q_weights')}")
    calls["n"] = 0
    mq.train()
    mq(b)
    print(f"quantum variant forward: QNode calls={calls['n']} (expect 3)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["b5", "b6diag", "b6fix", "b7"])
    a = ap.parse_args()
    {"b5": check_b5, "b6diag": check_b6diag, "b6fix": check_b6fix, "b7": check_b7}[a.check]()
