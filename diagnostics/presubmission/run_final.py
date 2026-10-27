#!/usr/bin/env python3
"""
Pre-submission fixed-epoch driver (Part C).

Protocol: every run trains exactly EPOCHS=50 epochs (patience=50, no early stop), fixed
validator (val pad 0, 768x960), per-epoch val ON, all approved fixes ON
(B1 attn_channel_fix, B2 aux_loss_batch_scale, B3 rem_fused_levels_only, B5 seeded_data_order,
B6 rgvqf_learned_encoder). REPORTED RESULT = last.pt at epoch 50, re-validated with the fixed
validator, written automatically to <out>/summary.csv at the end of each run.

Guarantees:
  - every training setting is passed explicitly (GRID below), and after the trainer is built
    (on_pretrain_routine_start) args.yaml is read back and asserted key-by-key -> abort on mismatch;
  - after the model is built (on_pretrain_routine_end) its modules are asserted against the config;
  - source snapshot + manifest (sha256 per file, pip freeze, git HEAD + working-tree diff hash,
    config, args, param_report) written into the run folder at start;
  - data come from diagnostics/presubmission/data/ (per-file symlinks; never writes into
    phase3/dataset_aligned);
  - copies of last.pt after epochs in KEEP_EPOCHS are kept as weights/epoch{N}.pt;
  - resumable: finished runs (last.pt epoch == -1) are skipped; partial runs resume WITH the
    custom trainer class (Ultralytics' bare resume would fall back to the stock trainer).

Usage:
  caffeinate -i -s .venv/bin/python -u diagnostics/presubmission/run_final.py --configs complete --seeds 0
  ... --configs all --seeds 0                 # 10 grid configs + classical control + 2 baselines
  ... --smoke --configs all                   # Part D: fraction 0.02, 1 epoch, into smoke_runs/
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import phase3.code.model.fusion.rgvqf as rgvqf_mod  # noqa: E402
from baseline_trainer import BaselineTrainer  # noqa: E402
from phase3.code.model.attention.cmaam import CMAAM  # noqa: E402
from phase3.code.model.cbam import CBAM  # noqa: E402
from phase3.code.model.fusion import FUSION_REGISTRY  # noqa: E402
from phase3.code.model.fusion.rgvqf import RGVQFFusion  # noqa: E402
from phase3.code.model.reliability.rem import REM  # noqa: E402
from phase3.code.model.yolov11_rgbt import RGBTDetectionModel  # noqa: E402
from phase3.code.rgbt_trainer import RGBTDetectionTrainer  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, EXTRA_ABLATIONS, make_trainer  # noqa: E402

EPOCHS = 12  # set from the canary's train/val-loss divergence (val box/dfl loss minimum at epoch 10 of 50)
KEEP_EPOCHS = (6, 12)
COLLAPSE_DROP = 0.10  # diagnostic only: in-loop mAP50-95 drop vs previous epoch that triggers a collapse_epoch{N}.pt copy
REPORTING_RULE = {
    "primary": "last.pt at the final epoch (12), re-validated with the fixed validator (val pad 0, 768x960)",
    "secondary": "mean of the in-loop mAP50-95 over the last 3 epochs (10-12)",
    "collapse": "a collapsed run is reported as-is with its collapse_flag; no run is ever re-run or replaced because of "
                "its result",
    "collapse_flag": "True if the final epoch's in-loop mAP50-95 is more than 0.10 below the mean of epochs 9-11",
}
# Campaign queue, in the pre-registered order: (1) baselines, (2) complete + cbam_add, (3) the other 9 fusion
# configs incl. classical control, (4) seeds 1..7 interleaved complete / cbam_add. All seed 0 unless stated.
FIXED_CONF = 0.25
DATA = {"fusion": f"{HERE}/data/fusion/data.yaml", "baseline_rgb": f"{HERE}/data/baseline_rgb/data.yaml",
        "baseline_ir": f"{HERE}/data/baseline_ir/data.yaml"}
FIXES = dict(attn_channel_fix=True, aux_loss_batch_scale=True, rem_fused_levels_only=True, seeded_data_order=True,
             rgvqf_learned_encoder=True)
# Every training-relevant setting, explicit. Identical to the step-5 grid except epochs/patience.
GRID = dict(epochs=EPOCHS, patience=EPOCHS, batch=4, imgsz=960, rect=True, deterministic=True, pretrained=True,
            optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
            warmup_momentum=0.8, warmup_bias_lr=0.1, nbs=64, cos_lr=False, box=10.0, cls=0.3, dfl=1.5,
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
            perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0, close_mosaic=0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            multi_scale=0.0, single_cls=False, fraction=1.0, cache=False, workers=0, val=True, plots=False,
            save_period=-1, freeze=None, dropout=0.0, amp=True, exist_ok=True)
ASSERT_KEYS = ["epochs", "patience", "batch", "imgsz", "rect", "deterministic", "optimizer", "lr0", "lrf", "box", "cls",
               "dfl", "mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "scale", "shear", "perspective",
               "flipud", "fliplr", "bgr", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "multi_scale", "fraction", "seed",
               "data", "val", "warmup_epochs", "nbs", "weight_decay", "momentum"]
SNAPSHOT_FILES = ["phase3/code/rgbt_trainer.py", "phase3/code/train_ablation_study.py",
                  "phase3/code/model/yolov11_rgbt.py", "phase3/code/model/rgbt_backbone_cbam.py",
                  "phase3/code/model/cbam.py", "phase3/code/model/attention/cmaam.py",
                  "phase3/code/model/reliability/rem.py", "phase3/code/model/fusion/__init__.py",
                  "phase3/code/model/fusion/add.py", "phase3/code/model/fusion/concat.py",
                  "phase3/code/model/fusion/qinspired.py", "phase3/code/model/fusion/rgvqf.py",
                  "phase3/code/losses/composite.py", "diagnostics/presubmission/baseline_trainer.py",
                  "diagnostics/presubmission/run_final.py", "diagnostics/presubmission/make_symlink_datasets.py"]

FREEZE_PATH = os.path.join(HERE, "CODE_FREEZE.json")


def code_hash():
    """Single hash over every source file that defines model / trainer / loss / data / driver behaviour."""
    h = hashlib.sha256()
    for rel in sorted(SNAPSHOT_FILES):
        h.update(rel.encode())
        h.update(open(os.path.join(ROOT, rel), "rb").read())
    return h.hexdigest()


def check_freeze(require):
    cur = code_hash()
    if os.path.isfile(FREEZE_PATH):
        frozen = json.load(open(FREEZE_PATH))["code_hash"]
        if cur != frozen:
            raise SystemExit(f"[freeze] ABORT: source changed since code freeze ({cur[:12]} != frozen {frozen[:12]})")
        print(f"[freeze] code hash matches freeze: {cur[:16]}")
    elif require:
        raise SystemExit("[freeze] ABORT: no CODE_FREEZE.json -- full runs require a frozen code hash")
    else:
        print(f"[freeze] no freeze yet (smoke/pre-freeze); current code hash {cur[:16]}")
    return cur


ALL_CONFIGS = {**ABLATIONS, **EXTRA_ABLATIONS}
BASELINES = {
    "baseline_thermal": {"name": "base_yolo11n_thermal", "data": "baseline_ir", "hsv": (0.0, 0.0, 0.0), "photometric_aug": False},
    "baseline_rgb": {"name": "base_yolo11n_rgb", "data": "baseline_rgb", "hsv": (0.015, 0.7, 0.4), "photometric_aug": True},
}
ORDER = list(ABLATIONS) + list(EXTRA_ABLATIONS) + list(BASELINES)


# ------------------------------------------------------------------ helpers
def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_settings(key, seed, smoke):
    s = dict(GRID, seed=seed)
    if key in BASELINES:
        b = BASELINES[key]
        s.update(data=DATA[b["data"]], hsv_h=b["hsv"][0], hsv_s=b["hsv"][1], hsv_v=b["hsv"][2])
    else:
        s["data"] = DATA["fusion"]
    if smoke:
        s.update(epochs=1, patience=1, fraction=0.02)
    return s


def trainer_class(key):
    if key in BASELINES:
        return type("FinalBaselineTrainer", (BaselineTrainer,),
                    {"seeded_data_order": True, "photometric_aug": BASELINES[key]["photometric_aug"]})
    return make_trainer({**ALL_CONFIGS[key], **FIXES})


def run_name(key, seed):
    return f"{(BASELINES.get(key) or ALL_CONFIGS[key])['name']}_s{seed}"


def finished(last_pt):
    if not os.path.isfile(last_pt):
        return False
    ck = torch.load(last_pt, map_location="cpu", weights_only=False)
    return isinstance(ck, dict) and ck.get("epoch", None) == -1


def expected_modules(key):
    """What the fusion backbone must contain for this config with FIXES on."""
    c = ALL_CONFIGS[key]
    att = c.get("attention") or ("cbam" if c.get("use_cbam", True) else "none")
    mode = c["fusion_mode"]
    modes = [mode, mode, mode] if c.get("multiscale", True) else ["add", "add", mode]
    rem = [c.get("use_reliability", False) and m == "rgvqf" for m in modes]  # B3 on
    return dict(attention=att, fusion=[FUSION_REGISTRY[m].__name__ for m in modes], rem=rem,
                entangle=[c.get("use_entanglement", True) if m == "rgvqf" else None for m in modes],
                learned_encoder=[m == "rgvqf" for m in modes],  # B6 on for every rgvqf level
                classical_ctrl=[bool(c.get("rgvqf_classical_ctrl")) and m == "rgvqf" for m in modes],
                alpha_rel=c.get("alpha_rel", 0.0), beta_q=c.get("beta_q", 0.0))


def actual_modules(model):
    fb = model.fusion_backbone
    fuses = [getattr(fb, f"fuse_{l}") for l in ("p3", "p4", "p5")]
    cb = [n for n, m in fb.named_modules() if isinstance(m, CBAM)]
    cm = [n for n, m in fb.named_modules() if isinstance(m, CMAAM)]
    return dict(attention="cbam" if cb else ("cmaam" if cm else "none"), fusion=[type(f).__name__ for f in fuses],
                rem=[isinstance(getattr(fb, f"rem_{l}", None), REM) for l in ("p3", "p4", "p5")],
                entangle=[f.entangle if isinstance(f, RGVQFFusion) else None for f in fuses],
                learned_encoder=[bool(getattr(f, "learned_encoder", False)) for f in fuses],
                classical_ctrl=[bool(getattr(f, "classical_ctrl", False)) for f in fuses],
                alpha_rel=model.alpha_rel, beta_q=model.beta_q,
                channel_fix=sorted({type(m.mlp[1]).__name__ for m in fb.modules() if isinstance(m, CBAM)} |
                                   {type(m.channel_mlp_rgb[1]).__name__ for m in fb.modules() if isinstance(m, CMAAM)}),
                aux_loss_batch_scale=model.aux_loss_batch_scale)


def write_manifest(run_dir, key, settings, extra=None):
    os.makedirs(run_dir, exist_ok=True)
    snap = os.path.join(run_dir, "source_snapshot")
    man_path = os.path.join(run_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {"starts": []}
    files = {}
    for rel in SNAPSHOT_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = sha256(src)
    if man.get("files") and man["files"] != files:
        man.setdefault("WARNING_source_changed_between_starts", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout  # noqa: E731
    man.update(files=files, config=BASELINES.get(key) or {**ALL_CONFIGS[key], **FIXES}, config_key=key,
               train_settings=settings, git_head=git("rev-parse", "HEAD").strip(),
               git_worktree_diff_sha256=hashlib.sha256(git("diff", "--", "phase3/code").encode()).hexdigest(),
               code_hash=code_hash(), encoder_gain_k=rgvqf_mod.ENCODER_GAIN_K, reporting_rule=REPORTING_RULE,
               python=sys.version, pip_freeze=subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                             text=True).stdout.splitlines())
    man["starts"].append(time.strftime("%Y-%m-%d %H:%M:%S"))
    if extra:
        man.update(extra)
    json.dump(man, open(man_path, "w"), indent=1, default=str)


def update_manifest(run_dir, **kw):
    p = os.path.join(run_dir, "manifest.json")
    m = json.load(open(p))
    m.update(kw)
    json.dump(m, open(p, "w"), indent=1, default=str)


# ------------------------------------------------------------------ callbacks
def make_callbacks(key, settings, run_dir, keep_epochs, probe):
    def assert_args(trainer):
        a = yaml.safe_load(open(os.path.join(trainer.save_dir, "args.yaml")))
        bad = {k: (a.get(k), settings[k]) for k in ASSERT_KEYS
               if not (a.get(k) == settings[k] or (isinstance(settings[k], float) and a.get(k) is not None
                                                    and abs(float(a[k]) - settings[k]) < 1e-12))}
        if bad:
            raise AssertionError(f"[{key}] args.yaml != intended settings: {bad}")
        update_manifest(run_dir, asserted_args={k: a.get(k) for k in ASSERT_KEYS})
        print(f"[assert] {key}: args.yaml matches {len(ASSERT_KEYS)} intended settings")

    def assert_model(trainer):
        m = trainer.model
        info = {}
        if key not in BASELINES:
            exp, act = expected_modules(key), actual_modules(m)
            cmp_keys = ["attention", "fusion", "rem", "entangle", "learned_encoder", "classical_ctrl", "alpha_rel", "beta_q"]
            bad = {k: (act[k], exp[k]) for k in cmp_keys if act[k] != exp[k]}
            if act["attention"] != "none" and act["channel_fix"] != ["LeakyReLU"]:
                bad["channel_fix"] = act["channel_fix"]
            if not act["aux_loss_batch_scale"]:
                bad["aux_loss_batch_scale"] = False
            if bad:
                raise AssertionError(f"[{key}] model modules != config: {bad}")
            info = dict(modules=act, param_report=m.param_report())
        else:
            if isinstance(m, RGBTDetectionModel):
                raise AssertionError("baseline built an RGBT model")
            info = dict(modules="stock DetectionModel", param_report={"used": sum(p.numel() for p in m.parameters())})
        update_manifest(run_dir, **info)
        print(f"[assert] {key}: modules match config; params {info['param_report']}")
        if probe is not None:
            probe["modules"], probe["params"] = info["modules"], info["param_report"]
            probe.install(trainer)

    def keep(trainer):
        e = trainer.epoch + 1
        if e in keep_epochs and os.path.isfile(trainer.last):
            shutil.copy2(trainer.last, os.path.join(trainer.wdir, f"epoch{e}.pt"))
        # Diagnostic collapse logging (results.csv already holds this epoch's row when on_model_save fires)
        csv_path = os.path.join(trainer.save_dir, "results.csv")
        if os.path.isfile(csv_path):
            rr = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(csv_path))]
            if len(rr) >= 2:
                prev, cur = float(rr[-2]["metrics/mAP50-95(B)"]), float(rr[-1]["metrics/mAP50-95(B)"])
                if prev - cur > COLLAPSE_DROP and os.path.isfile(trainer.last):
                    shutil.copy2(trainer.last, os.path.join(trainer.wdir, f"collapse_epoch{e}.pt"))
                    msg = f"[collapse] {key} epoch {e}: in-loop mAP50-95 {prev:.4f} -> {cur:.4f} (drop {prev - cur:.4f}); copied last.pt"
                    print(msg)
                    with open(os.path.join(trainer.save_dir, "collapse_log.txt"), "a") as fh:
                        fh.write(msg + "\n")

    return {"on_pretrain_routine_start": assert_args, "on_pretrain_routine_end": assert_model, "on_model_save": keep}


class SmokeProbe(dict):
    """Part D instrumentation (smoke runs only). Process-local CLASS-level wrappers -- nothing is attached to model
    instances, so nothing ends up pickled in checkpoints, and EMA/validation calls are seen too."""
    current = None
    _patched = False

    def install(self, trainer):
        self.update(qnode=0, qnode_at_train_end=None, fb_train=0, fb_eval=0, val_shapes=set(), losses=[], finite=True,
                    ir_identical=None)
        SmokeProbe.current = self
        trainer.add_callback("on_train_epoch_end", lambda t: SmokeProbe.current.update(
            qnode_at_train_end=SmokeProbe.current["qnode"], fb_train_at_train_end=SmokeProbe.current["fb_train"],
            train_batches=len(t.train_loader)))
        if not isinstance(trainer.model, RGBTDetectionModel):
            trainer.add_callback("on_train_batch_end", lambda t: SmokeProbe.current.__setitem__(
                "finite", SmokeProbe.current["finite"] and bool(torch.isfinite(t.loss).all())) or
                SmokeProbe.current["losses"].append([round(float(x), 5) for x in t.loss_items]))
        if SmokeProbe._patched:
            return
        SmokeProbe._patched = True
        from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

        from phase3.code.model.rgbt_backbone_cbam import RGBTBackboneCBAM
        orig_circ = rgvqf_mod._rgvqf_circuit

        def counted(*a, **k):
            SmokeProbe.current["qnode"] += 1
            return orig_circ(*a, **k)
        rgvqf_mod._rgvqf_circuit = counted

        orig_fb = RGBTBackboneCBAM.forward

        def fb_forward(self_, rgb, ir):
            p = SmokeProbe.current
            if self_.training:
                p["fb_train"] += 1
                if p["ir_identical"] is None:
                    p["ir_identical"] = bool((ir[:, 0] == ir[:, 1]).all() and (ir[:, 1] == ir[:, 2]).all())
            else:
                p["fb_eval"] += 1
                p["val_shapes"].add((tuple(rgb.shape[-2:]), tuple(ir.shape[-2:])))
            return orig_fb(self_, rgb, ir)
        RGBTBackboneCBAM.forward = fb_forward

        orig_loss = RGBTDetectionModel.loss

        def loss(self_, batch, preds=None):
            out = orig_loss(self_, batch, preds)
            if self_.training:
                v = out[0].detach().float().cpu()
                SmokeProbe.current["losses"].append([round(float(x), 5) for x in v])
                SmokeProbe.current["finite"] &= bool(torch.isfinite(v).all())
            return out
        RGBTDetectionModel.loss = loss

        orig_pb = DetectionTrainer.preprocess_batch  # baselines: check channels of the first train batch

        def preprocess_batch(self_, batch):
            b = orig_pb(self_, batch)
            p = SmokeProbe.current
            x = b["img"]
            if torch.is_tensor(x) and p["ir_identical"] is None and not isinstance(self_, RGBTDetectionTrainer):
                p["ir_identical"] = bool((x[:, 0] == x[:, 1]).all() and (x[:, 1] == x[:, 2]).all())
            return b
        DetectionTrainer.preprocess_batch = preprocess_batch

        orig_vp = DetectionValidator.preprocess  # baselines: val tensor shape (RGBT val shape comes from fb_forward)

        def vpre(self_, batch):
            b = orig_vp(self_, batch)
            if torch.is_tensor(b["img"]) and SmokeProbe.current is not None and type(self_).__name__ != "RGBTDetectionValidator":
                SmokeProbe.current["val_shapes"].add(tuple(b["img"].shape[-2:]))
            return b
        DetectionValidator.preprocess = vpre


# ------------------------------------------------------------------ revalidation (explicit settings)
def revalidate(key, weights, device, out_dir):
    """Same recipe as run_step5_retrain.revalidate_checkpoint (trainer.validate(), conf .001, iou .7, imgsz 960,
    rect, batch 4, P/R at conf 0.25) but with every setting explicit and the fixed-geometry trainer class."""
    pre = YOLO(weights).model
    base_cls = trainer_class(key)

    class Reval(base_cls):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pre

    s = run_settings(key, 0, smoke=False)
    s.update(epochs=1, conf=0.001, iou=0.7, project=os.path.join(out_dir, "_reval"), name=os.path.basename(
        os.path.dirname(os.path.dirname(weights))), model="yolo11n.pt", device=device)
    tr = Reval(overrides=s)
    tr._setup_train()
    tr.loss_items = torch.zeros(3, device=device)
    tr.epoch = 0
    tr.loss = torch.zeros(1, device=device)
    _, fitness = tr.validate()
    box = tr.validator.metrics.box
    idx = int(np.argmin(np.abs(box.px - FIXED_CONF)))
    res = dict(P=float(box.p_curve[0][idx]), R=float(box.r_curve[0][idx]), mAP50=float(box.map50),
               mAP50_95=float(box.map), fitness=float(fitness) if fitness is not None else None)
    del tr, pre
    if device == "mps":
        torch.mps.empty_cache()
    return res


# ------------------------------------------------------------------ A2 table (RGVQF configs)
def a2_table(key, weights, n_images=100):
    """Per scale over a FIXED RANDOM subset of n val images (seed 0; not consecutive near-identical frames):
    <Z> per-qubit std, gate std, |x|>0.9pi fraction, encoder BN stats. Eval mode at batch 1, and whole-model train
    mode in SHUFFLED batches of 4 over the same subset. Fresh checkpoint load per mode; nothing saved."""
    from torch.utils.data import DataLoader, Subset
    out = {}
    for label, bs, mode in (("eval_b1", 1, "eval"), ("train_b4", 4, "train")):
        tr = trainer_class(key)(overrides=dict(run_settings(key, 0, False), device="cpu", model="yolo11n.pt",
                                               project=os.path.join(HERE, "_a2_scratch"), name="a2"))
        m = YOLO(weights).model.float()
        tr.model = m
        tr.set_model_attributes()
        m.train() if mode == "train" else m.eval()
        caps = {}
        for lvl in ("p3", "p4", "p5"):
            f = getattr(m.fusion_backbone, f"fuse_{lvl}")
            if isinstance(f, RGVQFFusion):
                orig = f._descriptor
                f._descriptor = lambda *a, _o=orig, _l=lvl, **k: caps.setdefault(_l, {"x": [], "q": [], "g": []})["x"].append(
                    _o(*a, **k).detach().float().cpu()) or caps[_l]["x"][-1]
        ds = tr.build_dataset(tr.data["val"], mode="val", batch=bs)
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n_images].tolist()
        dl = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=(mode == "train"), collate_fn=ds.collate_fn,
                        generator=torch.Generator().manual_seed(0))
        tr.test_loader = dl
        v = tr.get_validator()
        v.device = tr.device
        v.args.half = False
        with torch.no_grad():
            for b in dl:
                m(v.preprocess(b)["img"])
                for lvl in caps:
                    f = getattr(m.fusion_backbone, f"fuse_{lvl}")
                    caps[lvl]["q"].append(f.last_q_out.float().cpu())
                    caps[lvl]["g"].append(torch.sigmoid(f.to_gate(f.last_q_out.float())).cpu())
        for lvl, c in caps.items():
            x, q, g = (torch.cat(c[k]) for k in ("x", "q", "g"))
            f = getattr(m.fusion_backbone, f"fuse_{lvl}")
            bn = getattr(f, "encoder_bn", None)
            out.setdefault(lvl, {})[label] = dict(
                n_images=int(x.shape[0]), mean_qubit_std=float(q.std(0).mean()), gate_mean=float(g.mean()),
                gate_std_mean=float(g.std(0).mean()), gate_std_max=float(g.std(0).max()),
                x_sat_frac=float((x.abs() > 0.9 * torch.pi).float().mean()), x_std_mean=float(x.std(0).mean()),
                encoder_k=getattr(f, "encoder_k", None),
                bn_running_var=None if bn is None else [round(float(t), 5) for t in bn.running_var],
                bn_running_mean=None if bn is None else [round(float(t), 4) for t in bn.running_mean],
                # learned BN affine: does training scale gamma up to compensate for encoder_k?
                bn_gamma=None if bn is None else [round(float(t), 4) for t in bn.weight],
                bn_beta=None if bn is None else [round(float(t), 4) for t in bn.bias],
                effective_gain=None if bn is None else round(float(bn.weight.abs().mean()) / getattr(f, "encoder_k", 1.0), 4))
    return out


# ------------------------------------------------------------------ one run
def run_one(key, seed, out_root, device, smoke, do_a2):
    name = run_name(key, seed)
    runs = os.path.join(out_root, "runs")
    run_dir = os.path.join(runs, name)
    last = os.path.join(run_dir, "weights", "last.pt")
    settings = run_settings(key, seed, smoke)
    keep_epochs = (1,) if smoke else KEEP_EPOCHS
    probe = SmokeProbe() if smoke else None
    cls = trainer_class(key)
    t0 = time.time()
    if finished(last):
        print(f"[{name}] finished already -- skipping training")
    else:
        write_manifest(run_dir, key, settings)
        cbs = make_callbacks(key, settings, run_dir, keep_epochs, probe)
        if os.path.isfile(last):
            print(f"[{name}] resuming {last} with {cls.__name__}")
            y = YOLO(last)
            for ev, fn in cbs.items():
                y.add_callback(ev, fn)
            y.train(trainer=cls, resume=True)
        else:
            print(f"[{name}] fresh training -> {run_dir}")
            y = YOLO("yolo11n.pt")
            for ev, fn in cbs.items():
                y.add_callback(ev, fn)
            y.train(trainer=cls, project=runs, name=name, device=device, **settings)
    train_s = time.time() - t0

    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(os.path.join(run_dir, "results.csv")))]
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    peak = int(np.argmax(m5095))
    mean_last3 = float(np.mean(m5095[-3:]))
    collapse_flag = bool(len(m5095) >= 4 and m5095[-1] < float(np.mean(m5095[-4:-1])) - COLLAPSE_DROP)
    reval = revalidate(key, last, device, out_root)
    row = dict(run=name, config=key, seed=seed, epochs_trained=len(rows), reported_ckpt="last.pt",
               mAP50=round(reval["mAP50"], 5), mAP50_95=round(reval["mAP50_95"], 5), P_at_0_25=round(reval["P"], 4),
               R_at_0_25=round(reval["R"], 4), inloop_last_mAP50_95=round(m5095[-1], 5),
               mean_last3_mAP50_95=round(mean_last3, 5), collapse_flag=collapse_flag,
               inloop_peak_epoch=int(float(rows[peak]["epoch"])), inloop_peak_mAP50_95=round(m5095[peak], 5),
               train_wall_s=round(train_s), sec_per_epoch=round(float(rows[-1]["time"]) / len(rows), 1))
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    row.update(params_used=man.get("param_report", {}).get("used"), params_quantum=man.get("param_report", {}).get("quantum", 0))
    summ = os.path.join(out_root, "summary.csv")
    old = [r for r in csv.DictReader(open(summ))] if os.path.isfile(summ) else []
    old = [r for r in old if r["run"] != name] + [row]
    with open(summ, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(old)
    json.dump({"per_epoch": [(int(float(r["epoch"])), float(r["metrics/mAP50(B)"]), float(r["metrics/mAP50-95(B)"]))
                             for r in rows], "reval_last": reval}, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    print(f"[{name}] RESULT (last.pt, fixed validator): {row}")

    if do_a2 and key not in BASELINES and ALL_CONFIGS[key]["fusion_mode"] == "rgvqf":
        a2 = {}
        for e in keep_epochs:
            p = os.path.join(run_dir, "weights", f"epoch{e}.pt")
            if os.path.isfile(p):
                a2[f"epoch{e}"] = a2_table(key, p, n_images=100 if not smoke else 8)
        json.dump(a2, open(os.path.join(run_dir, "a2.json"), "w"), indent=1)
        print(f"[{name}] A2 table at epochs {list(a2)} -> a2.json")
    return row, (dict(probe) if probe is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help=f"keys or 'all': {ORDER}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true", help="Part D: fraction 0.02, 1 epoch, into smoke_runs/")
    ap.add_argument("--no-a2", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--freeze", action="store_true", help="write CODE_FREEZE.json with the current code hash and exit")
    ap.add_argument("--refreeze", metavar="REASON", help="re-freeze: move the current freeze into history, record old/new hash")
    ap.add_argument("--plan", choices=["campaign"], help="run the pre-registered campaign queue (overrides --configs/--seeds)")
    a = ap.parse_args()
    if a.freeze:
        if os.path.isfile(FREEZE_PATH):
            raise SystemExit(f"[freeze] {FREEZE_PATH} already exists -- refusing to overwrite")
        json.dump({"code_hash": code_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {
            rel: sha256(os.path.join(ROOT, rel)) for rel in SNAPSHOT_FILES}, "encoder_gain_k": rgvqf_mod.ENCODER_GAIN_K},
            open(FREEZE_PATH, "w"), indent=1)
        print(f"[freeze] wrote {FREEZE_PATH}: {code_hash()}")
        return
    if a.refreeze:
        old = json.load(open(FREEZE_PATH))
        new_hash = code_hash()
        if new_hash == old["code_hash"]:
            raise SystemExit("[freeze] refreeze requested but code hash is unchanged")
        hist = old.pop("history", [])
        hist.append({**old, "superseded": time.strftime("%Y-%m-%d %H:%M:%S")})
        json.dump({"code_hash": new_hash, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "previous_hash": old["code_hash"],
                   "reason": a.refreeze, "epochs": EPOCHS, "keep_epochs": list(KEEP_EPOCHS),
                   "encoder_gain_k": rgvqf_mod.ENCODER_GAIN_K, "reporting_rule": REPORTING_RULE,
                   "files": {rel: sha256(os.path.join(ROOT, rel)) for rel in SNAPSHOT_FILES}, "history": hist},
                  open(FREEZE_PATH, "w"), indent=1)
        print(f"[freeze] re-frozen: {old['code_hash'][:12]} -> {new_hash[:12]} ({a.refreeze})")
        return
    check_freeze(require=not a.smoke)
    keys = ORDER if a.configs == ["all"] else a.configs
    for k in keys:
        assert k in ORDER, f"unknown config {k}"
    import inspect
    assert "pad=0.0" in inspect.getsource(RGBTDetectionTrainer.build_dataset), "fixed validator missing -- STOP"
    assert "pad=0.0" in inspect.getsource(BaselineTrainer.build_dataset), "baseline fixed validator missing -- STOP"
    for p in DATA.values():
        assert os.path.isfile(p), f"missing {p} -- run make_symlink_datasets.py"
    # Output root is tied to EPOCHS so the 50-epoch canary (final_runs/) is never resumed, skipped or pooled.
    out_root = os.path.join(HERE, f"smoke_runs_e{EPOCHS}" if a.smoke else f"campaign_e{EPOCHS}")
    os.makedirs(out_root, exist_ok=True)
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.plan == "campaign":
        rest = [k for k in list(ABLATIONS) + list(EXTRA_ABLATIONS) if k not in ("complete", "cbam_add")]
        queue = [("baseline_thermal", 0), ("baseline_rgb", 0), ("complete", 0), ("cbam_add", 0)] + [(k, 0) for k in rest]
        queue += [(k, s) for s in range(1, 8) for k in ("complete", "cbam_add")]
    else:
        queue = [(k, s) for s in a.seeds for k in keys]
    print(f"[plan] {len(queue)} runs: {[f'{k}:s{s}' for k, s in queue]}")
    probe_path = os.path.join(out_root, "smoke_probes.json")
    probes = json.load(open(probe_path)) if os.path.isfile(probe_path) else {}  # merge across processes
    for k, seed in queue:
        row, probe = run_one(k, seed, out_root, device, a.smoke, not a.no_a2)
        if probe is not None:
            probe["val_shapes"] = sorted(map(str, probe["val_shapes"]))
            probe["losses"] = probe["losses"][:2] + probe["losses"][-1:]
            probes[run_name(k, seed)] = {"row": row, "probe": dict(probe)}
            json.dump(probes, open(probe_path, "w"), indent=1, default=str)
    print("[done]")


if __name__ == "__main__":
    main()
