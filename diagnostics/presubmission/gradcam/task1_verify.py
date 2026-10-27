#!/usr/bin/env python3
"""
Task 1 (read-only): verify attention init, campaign_e12 training settings, and trainable params.

Writes task1_report.json + task1_report.md next to this file. Touches nothing else.
  .venv/bin/python diagnostics/presubmission/gradcam/task1_verify.py
"""
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
PRE = os.path.join(ROOT, "diagnostics/presubmission")
CAMP = os.path.join(PRE, "campaign_e12")
sys.path.insert(0, ROOT)
sys.path.insert(0, PRE)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from phase3.code.model.attention.cmaam import CMAAM  # noqa: E402
from phase3.code.model.cbam import CBAM  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# ------------------------------------------------------------------ 0. freeze check
def freeze_check():
    f = json.load(open(os.path.join(PRE, "CODE_FREEZE.json")))
    files = {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() == h for p, h in f["files"].items()}
    h = hashlib.sha256()
    for rel in sorted(f["files"]):
        h.update(rel.encode())
        h.update(open(os.path.join(ROOT, rel), "rb").read())
    return {"frozen_code_hash": f["code_hash"], "current_code_hash": h.hexdigest(),
            "match": h.hexdigest() == f["code_hash"] and all(files.values()), "per_file_match": files}


# ------------------------------------------------------------------ 1a. attention init
def describe_attn(m):
    if isinstance(m, CBAM):
        mlps, spatial = {"mlp": m.mlp}, {"spatialConv": m.spatialConv}
    else:
        mlps = {"channel_mlp_rgb": m.channel_mlp_rgb, "channel_mlp_ir": m.channel_mlp_ir}
        spatial = {"spatial_conv_rgb": m.spatial_conv_rgb, "spatial_conv_ir": m.spatial_conv_ir}
    out = {}
    for k, mlp in mlps.items():
        act = mlp[1]
        w = mlp[2].weight.detach().float()
        out[k] = {"activation": type(act).__name__,
                  "negative_slope": getattr(act, "negative_slope", None),
                  "last_layer_std": float(w.std()), "last_layer_absmax": float(w.abs().max()),
                  "last_layer_all_zero": bool((w == 0).all())}
    for k, conv in spatial.items():
        w = conv.weight.detach().float()
        out[k] = {"kernel": list(conv.kernel_size), "all_zero": bool((w == 0).all()), "std": float(w.std())}
    out["any_bias"] = any(getattr(c, "bias", None) is not None for c in m.modules() if isinstance(c, nn.Conv2d))
    return out


def attention_init():
    """Build fresh models exactly as the campaign trainer does (FIXES on) and read the init back."""
    from run_final import ALL_CONFIGS, FIXES, trainer_class  # noqa: F401
    from phase3.code.model.yolov11_rgbt import RGBTDetectionModel

    res = {}
    for key, att in (("cbam_add", "cbam"), ("complete", "cmaam")):
        c = ALL_CONFIGS[key]
        torch.manual_seed(0)
        model = RGBTDetectionModel(cfg="yolo11n.yaml", nc=1, fusion_mode=c["fusion_mode"], use_cbam=c.get("use_cbam", True),
                                   attention=c.get("attention"), use_reliability=c.get("use_reliability", False),
                                   verbose=False, attn_channel_fix=FIXES["attn_channel_fix"],
                                   rem_fused_levels_only=FIXES["rem_fused_levels_only"],
                                   rgvqf_learned_encoder=FIXES["rgvqf_learned_encoder"] and c["fusion_mode"] == "rgvqf")
        mods = {n: describe_attn(m) for n, m in model.fusion_backbone.named_modules() if isinstance(m, (CBAM, CMAAM))}
        res[key] = {"attention": att, "fresh_init": mods}
    # constructor defaults (the value channel_fix=True uses)
    import inspect
    res["defaults"] = {"CBAM": str(inspect.signature(CBAM.__init__)), "CMAAM": str(inspect.signature(CMAAM.__init__))}
    return res


def attention_trained(run):
    ck = torch.load(os.path.join(CAMP, "runs", run, "weights/last.pt"), map_location="cpu", weights_only=False)
    m = ck["ema"] or ck["model"]
    mods = [mm for mm in m.fusion_backbone.modules() if isinstance(mm, (CBAM, CMAAM))]
    acts = sorted({type(mm.mlp[1] if isinstance(mm, CBAM) else mm.channel_mlp_rgb[1]).__name__ for mm in mods})
    slopes = sorted({getattr(mm.mlp[1] if isinstance(mm, CBAM) else mm.channel_mlp_rgb[1], "negative_slope", None)
                     for mm in mods}, key=str)
    biases = [n for n, p in m.fusion_backbone.named_parameters()
              if n.endswith(".bias") and any(t in n for t in ("cbam", "cmaam"))]
    return {"n_modules": len(mods), "activations": acts, "slopes": slopes, "attention_bias_params": biases}


# ------------------------------------------------------------------ 1b. args + logs
KEYS = ["batch", "nbs", "optimizer", "lr0", "warmup_epochs", "lrf", "epochs", "patience", "data", "device", "seed"]


def split_logs():
    """Per run: list of sections (fresh/resume) with optimizer lines, device lines, scan counts."""
    runs = {}
    for logf in ("campaign.log", "thermal_seeds.log"):
        cur = None
        segs = (s for raw in open(os.path.join(CAMP, logf), errors="replace") for s in ANSI.sub("", raw).split("\r"))
        for line in segs:
            line = line.rstrip("\n")
            mm =re.match(r"^\[(\w+_s\d)\] (fresh training|resuming|finished already)", line)
            if mm:
                if mm.group(2) == "finished already":
                    runs.setdefault(mm.group(1), []).append({"log": logf, "kind": "skipped (finished)"})
                    cur = None
                    continue
                cur = {"log": logf, "kind": mm.group(2), "optimizer": [], "device": set(), "train_imgs": set(),
                       "val_imgs": set(), "data_dirs": set()}
                runs.setdefault(mm.group(1), []).append(cur)
                continue
            if cur is None:
                continue
            if re.match(r"^\[\w+_s\d\] RESULT", line):
                cur = None
                continue
            if "optimizer:" in line and "with parameter groups" in line:
                cur["optimizer"].append(line.split("optimizer:")[1].strip())
            m2 = re.search(r"Ultralytics [\d.]+ .*?torch-[\d.]+ (.*)$", line)
            if m2:
                cur["device"].add(m2.group(1).strip())
            m3 = re.search(r"(train|val): Scanning (\S+?)\.cache\.\.\. (\d+) images", line)
            if m3:
                cur[f"{m3.group(1)}_imgs"].add(int(m3.group(3)))
                cur["data_dirs"].add(m3.group(2).rsplit("/labels", 1)[0])
    for v in runs.values():
        for s in v:
            for k in ("device", "train_imgs", "val_imgs", "data_dirs"):
                if k in s:
                    s[k] = sorted(s[k])
    return runs


def settings_table():
    logs = split_logs()
    rows = []
    for d in sorted(glob.glob(os.path.join(CAMP, "runs/*/"))):
        run = os.path.basename(d.rstrip("/"))
        a = yaml.safe_load(open(os.path.join(d, "args.yaml")))
        secs = [s for s in logs.get(run, []) if s["kind"] != "skipped (finished)"]
        opt = sorted({o for s in secs for o in s["optimizer"]})
        ck = torch.load(os.path.join(d, "weights/last.pt"), map_location="cpu", weights_only=False)
        rows.append({"run": run, **{k: a.get(k) for k in KEYS},
                     "optimizer_resolved": [re.sub(r" with parameter groups.*", "", o) for o in opt],
                     "param_groups": [re.sub(r".*with parameter groups ", "", o) for o in opt],
                     "train_images": sorted({n for s in secs for n in s["train_imgs"]}),
                     "val_images": sorted({n for s in secs for n in s["val_imgs"]}),
                     "data_dirs_scanned": sorted({n for s in secs for n in s["data_dirs"]}),
                     "device_log": sorted({n for s in secs for n in s["device"]}),
                     "sections": [s["kind"] for s in logs.get(run, [])],
                     "last_pt_epoch": ck.get("epoch"), "last_pt_has_ema": ck.get("ema") is not None})
    return rows


def flag_rows(rows):
    """Flag anything that differs from the modal value (data/val counts compared within modality)."""
    from collections import Counter
    flags = []
    for k in ["batch", "nbs", "optimizer", "lr0", "warmup_epochs", "lrf", "epochs", "patience", "device"]:
        mode = Counter(json.dumps(r[k]) for r in rows).most_common(1)[0][0]
        flags += [f"{r['run']}: {k}={r[k]} (others {mode})" for r in rows if json.dumps(r[k]) != mode]
    for k in ["optimizer_resolved", "train_images", "val_images", "device_log"]:
        mode = Counter(json.dumps(r[k]) for r in rows).most_common(1)[0][0]
        flags += [f"{r['run']}: {k}={r[k]} (others {mode})" for r in rows if json.dumps(r[k]) != mode]
    for r in rows:
        exp = "baseline_ir" if "thermal" in r["run"] else "baseline_rgb" if "_rgb_" in r["run"] else "fusion"
        if f"/data/{exp}/" not in r["data"]:
            flags.append(f"{r['run']}: data={r['data']} (expected {exp})")
        if r["sections"].count("fresh training") + r["sections"].count("resuming") != 1:
            flags.append(f"{r['run']}: training log sections = {r['sections']} (interrupted + resumed)")
        if r["last_pt_epoch"] != -1:
            flags.append(f"{r['run']}: last.pt epoch={r['last_pt_epoch']} (not marked finished)")
    return flags


# ------------------------------------------------------------------ 1c. params
def params():
    out = {}
    for d in sorted(glob.glob(os.path.join(CAMP, "runs/*_s0/"))):
        run = os.path.basename(d.rstrip("/"))
        ck = torch.load(os.path.join(d, "weights/last.pt"), map_location="cpu", weights_only=False)
        m = ck["ema"] or ck["model"]
        total = sum(p.numel() for p in m.parameters())
        if hasattr(m, "fusion_backbone"):
            unused = sum(p.numel() for i in range(11) for p in m.model[i].parameters())
            q = sum(p.numel() for n, p in m.fusion_backbone.named_parameters() if n.endswith("q_weights"))
        else:
            unused, q = 0, 0
        # Trainable = what the optimizer could update. The checkpoint stores requires_grad=False
        # (EMA/half), so count all float params; frozen layers would be named in args.freeze (None).
        out[run] = {"total_in_ckpt": total, "unused_stock_backbone": unused, "trainable_used": total - unused,
                    "quantum": q, "freeze_arg": yaml.safe_load(open(os.path.join(d, "args.yaml"))).get("freeze")}
    return out


def main():
    rep = {"freeze": freeze_check()}
    rep["attention_init"] = attention_init()
    rep["attention_trained"] = {r: attention_trained(r) for r in
                                ("abl_complete_s0", "abl_cbam_no_vqc_s0", "abl_cmaam_add_s0", "abl_cbam_rgvqf_rem_s0")}
    rows = settings_table()
    rep["settings"] = rows
    rep["settings_flags"] = flag_rows(rows)
    rep["params"] = params()
    json.dump(rep, open(os.path.join(HERE, "task1_report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("settings_flags", "params")}, indent=1, default=str))


if __name__ == "__main__":
    main()
