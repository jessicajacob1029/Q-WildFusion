#!/usr/bin/env python3
"""
Part D: PASS/FAIL table for the smoke runs (diagnostics/presubmission/smoke_runs/), plus disk-space check.
    .venv/bin/python -I diagnostics/presubmission/partD_table.py
"""
import json
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SMOKE = os.path.join(HERE, "smoke_runs")
AUG_KEYS = ["mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "box", "cls", "dfl", "rect", "imgsz"]
EXPECT_Q = {"abl_no_cbam_add": 0, "abl_cbam_no_vqc": 0, "abl_cbam_vqc_proposed": 0, "abl_cbam_rgvqf_proposed": 3,
            "abl_cbam_rgvqf_no_entangle": 3, "abl_cbam_rgvqf_rem": 3, "abl_cmaam_add": 0, "abl_cmaam_rgvqf_rem": 3,
            "abl_mqfp_single_scale": 1, "abl_complete": 3, "abl_cbam_rgvqf_classical_ctrl": 0,
            "base_yolo11n_thermal": 0, "base_yolo11n_rgb": 0}

sys.argv = [sys.argv[0]]
import run_final as rf  # noqa: E402

FINAL_HASH = rf.code_hash()
probes = json.load(open(os.path.join(SMOKE, "smoke_probes.json")))
L = ["| run | modules (asserted vs config) | params used / quantum | QNode per train fwd (exp) | losses finite | aux terms (α·L_rel, β·L_q) "
     "| val tensor | IR 3 identical ch | metrics | snapshot+manifest | ckpt in new folder | asserted aug settings | code hash | verdict |\n",
     "|" + "---|" * 14 + "\n"]
allpass = True
for name, exp_q in EXPECT_Q.items():
    run = os.path.join(SMOKE, "runs", f"{name}_s0")
    if name + "_s0" not in probes:
        L.append(f"| {name} | MISSING | | | | | | | | | | | | **FAIL** |\n")
        allpass = False
        continue
    p = probes[name + "_s0"]["probe"]
    man = json.load(open(os.path.join(run, "manifest.json")))
    args = yaml.safe_load(open(os.path.join(run, "args.yaml")))
    mods = man.get("modules")
    mod_s = mods if isinstance(mods, str) else (
        f"{mods['attention']}; {'/'.join(m.replace('Fusion', '') for m in mods['fusion'])}; REM {''.join('Y' if r else '-' for r in mods['rem'])}; "
        f"enc {''.join('Y' if r else '-' for r in mods['learned_encoder'])}; ctrl {''.join('Y' if r else '-' for r in mods['classical_ctrl'])}; "
        f"att-act {','.join(mods['channel_fix']) or '-'}")
    pr = man.get("param_report", {})
    ntrain = p.get("fb_train_at_train_end") or 0
    qpf = (p.get("qnode_at_train_end") or 0) / ntrain if ntrain else 0.0
    is_base = name.startswith("base_")
    q_ok = (qpf == exp_q) if not is_base else (p.get("qnode_at_train_end", 0) in (0, None))
    losses = p["losses"]
    aux = [l[3:5] for l in losses if len(l) == 5]
    aux_nonzero = any(abs(v) > 0 for a in aux for v in a)
    aux_ok = aux_nonzero == (name == "abl_complete") if not is_base else True
    val = p["val_shapes"]
    val_ok = all("768, 960" in v for v in val) and len(val) > 0
    ir_ok = p.get("ir_identical") is True if (not is_base or "thermal" in name) else True
    metrics_ok = os.path.isfile(os.path.join(run, "results.csv")) and os.path.isfile(os.path.join(run, "result.json"))
    snap_ok = os.path.isdir(os.path.join(run, "source_snapshot")) and len(man.get("files", {})) >= 15
    ck_ok = all(os.path.isfile(os.path.join(run, "weights", f)) for f in ("last.pt", "epoch1.pt"))
    asserted = man.get("asserted_args", {})
    aug_ok = all(k in asserted for k in ("mosaic", "fliplr", "box")) and all(args[k] == asserted.get(k, args[k]) for k in AUG_KEYS)
    aug_s = ", ".join(f"{k}={args[k]}" for k in AUG_KEYS)
    hash_ok = man.get("code_hash") == FINAL_HASH
    ok = all([q_ok, p["finite"], aux_ok, val_ok, ir_ok, metrics_ok, snap_ok, ck_ok, aug_ok, mods is not None, hash_ok])
    allpass &= ok
    aux_s = "n/a" if is_base else ("present " + str(aux[0]) if aux_nonzero else "0, 0")
    L.append(f"| {name} | {mod_s} | {pr.get('used')} / {pr.get('quantum', 0)} | {qpf:.2f} ({exp_q}) | {p['finite']} | {aux_s} | "
             f"{'; '.join(val)} | {p.get('ir_identical') if (not is_base or 'thermal' in name) else 'n/a (RGB)'} | {metrics_ok} | {snap_ok} | {ck_ok} | "
             f"{aug_s} | {(man.get('code_hash') or 'none')[:12]}{'' if hash_ok else ' (STALE)'} | **{'PASS' if ok else 'FAIL'}** |\n")

# disk
du = shutil.disk_usage(HERE)
per_run_mb = max(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(SMOKE, "runs", r)) for f in fs)
                 for r in os.listdir(os.path.join(SMOKE, "runs"))) / 1e6
L.append(f"\nDisk: free {du.free / 1e9:.0f} GB. Largest smoke run folder {per_run_mb:.0f} MB (last/best/epoch1 + snapshot). "
         f"A 50-epoch run keeps last, best, epoch10/25/50 (~5 checkpoints) ≈ {per_run_mb * 5 / 3:.0f} MB; 40 runs ≈ "
         f"{per_run_mb * 5 / 3 * 40 / 1000:.1f} GB → {'OK' if du.free / 1e9 > 50 else 'LOW'}.\n")
L.append(f"\nFinal code hash: `{FINAL_HASH}`\n")
L.append(f"\n**Overall: {'PASS' if allpass else 'FAIL'}**\n")
open(os.path.join(HERE, "partD_report.md"), "w").write("".join(L))
print("".join(L))
