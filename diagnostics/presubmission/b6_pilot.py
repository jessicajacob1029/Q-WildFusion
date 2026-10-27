#!/usr/bin/env python3
"""
B6 pilot: cbam_rgvqf, seed 0, 5 epochs (patience 50), fixed validator (val pad 0),
B1+B2+B3+B5 ON, trained twice: B6 (rgvqf_learned_encoder) OFF and ON.

At the end of each run: epoch-5 metrics (results.csv, in-loop EMA = last.pt),
re-validation of last.pt via run_step5_retrain.revalidate_checkpoint (P/R @ conf 0.25),
and the A2 table on last.pt (per-qubit <Z> std and gate std across 100 val images, per scale).
Writes diagnostics/presubmission/b6_pilot/summary.md.

Resumable: a finished run (last.pt epoch == -1) is skipped; a partial run is resumed
WITH the custom trainer class (Ultralytics' resume path otherwise falls back to the
stock DetectionTrainer, which would feed bare RGB).

    caffeinate -i -s .venv/bin/python -u diagnostics/presubmission/b6_pilot.py 2>&1 | tee -a diagnostics/presubmission/b6_pilot/run.log
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import run_step5_retrain as rsr  # noqa: E402
from phase3.code.model.fusion.rgvqf import RGVQFFusion  # noqa: E402
from phase3.code.train_ablation_study import ABLATIONS, make_trainer  # noqa: E402

OUT = os.path.join(ROOT, "diagnostics", "presubmission", "b6_pilot")
RUNS = os.path.join(OUT, "runs")
EPOCHS = 5
SEED = 0
BASE_FLAGS = dict(attn_channel_fix=True, aux_loss_batch_scale=True, rem_fused_levels_only=True, seeded_data_order=True)
VARIANTS = {"b6_off": dict(BASE_FLAGS, rgvqf_learned_encoder=False),
            "b6_on": dict(BASE_FLAGS, rgvqf_learned_encoder=True)}
SNAPSHOT_FILES = ["phase3/code/rgbt_trainer.py", "phase3/code/train_ablation_study.py",
                  "phase3/code/model/yolov11_rgbt.py", "phase3/code/model/rgbt_backbone_cbam.py",
                  "phase3/code/model/cbam.py", "phase3/code/model/attention/cmaam.py",
                  "phase3/code/model/reliability/rem.py", "phase3/code/model/fusion/__init__.py",
                  "phase3/code/model/fusion/add.py", "phase3/code/model/fusion/concat.py",
                  "phase3/code/model/fusion/qinspired.py", "phase3/code/model/fusion/rgvqf.py",
                  "phase3/code/losses/composite.py", "run_step5_retrain.py", __file__]
TRAIN_ARGS = dict(data=rsr.DATA_YAML, epochs=EPOCHS, imgsz=960, batch=4, workers=0, fraction=1.0, patience=50,
                  seed=SEED, deterministic=True, pretrained=True, box=10.0, cls=0.3, dfl=1.5, rect=True, mosaic=0.0,
                  mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                  flipud=0.0, fliplr=0.0, close_mosaic=0, val=True, plots=False, exist_ok=True, project=RUNS)
rsr.OUT_DIR = OUT  # revalidate_checkpoint scratch -> b6_pilot/_scratch_runs


def write_snapshot(run_dir, config):
    snap = os.path.join(run_dir, "source_snapshot")
    os.makedirs(snap, exist_ok=True)
    manifest = {"files": {}, "config": config, "train_args": TRAIN_ARGS, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for f in SNAPSHOT_FILES:
        src = os.path.join(ROOT, f) if not os.path.isabs(f) else f
        rel = os.path.relpath(src, ROOT)
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        manifest["files"][rel] = hashlib.sha256(open(src, "rb").read()).hexdigest()
    manifest["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=1)


def finished(last_pt):
    if not os.path.isfile(last_pt):
        return False
    ck = torch.load(last_pt, map_location="cpu", weights_only=False)
    return isinstance(ck, dict) and ck.get("epoch", None) == -1


def train(name, flags, device):
    config = {**ABLATIONS["cbam_rgvqf"], **flags}
    cls = make_trainer(config)
    run_dir = os.path.join(RUNS, name)
    last = os.path.join(run_dir, "weights", "last.pt")
    if finished(last):
        print(f"[{name}] finished already -- skipping training")
        return run_dir
    os.makedirs(run_dir, exist_ok=True)
    write_snapshot(run_dir, config)
    if os.path.isfile(last):
        print(f"[{name}] resuming {last} with the custom trainer")
        YOLO(last).train(trainer=cls, resume=True)
    else:
        print(f"[{name}] fresh training -> {run_dir}  flags={flags}")
        YOLO("yolo11n.pt").train(trainer=cls, name=name, device=device, **TRAIN_ARGS)
    return run_dir


def a2_table(last_pt, flags):
    """per scale: per-qubit <Z> std and gate std across 100 val images (fixed geometry), last.pt, eval mode."""
    tr = make_trainer({**ABLATIONS["cbam_rgvqf"], **flags})(overrides=dict(
        data=rsr.DATA_YAML, imgsz=960, batch=4, device="cpu", workers=0, rect=True, model="yolo11n.pt",
        project=os.path.join(OUT, "_scratch"), name="a2", exist_ok=True, plots=False))
    m = YOLO(last_pt).model.float().eval()
    tr.model = m
    tr.set_model_attributes()
    dl = tr.get_dataloader(tr.data["val"], batch_size=4, rank=-1, mode="val")
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    qs, gs = {}, {}
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= 25:
                break
            m(v.preprocess(b)["img"])
            for lvl in ("p3", "p4", "p5"):
                f = getattr(m.fusion_backbone, f"fuse_{lvl}")
                if isinstance(f, RGVQFFusion):
                    q = f.last_q_out.float()
                    qs.setdefault(lvl, []).append(q)
                    gs.setdefault(lvl, []).append(torch.sigmoid(f.to_gate(q)))
    rows = {}
    for lvl in qs:
        q, g = torch.cat(qs[lvl]), torch.cat(gs[lvl])
        rows[lvl] = dict(qubit_std=[round(float(x), 4) for x in q.std(0)], mean_qubit_std=float(q.std(0).mean()),
                         gate_mean=float(g.mean()), gate_std_mean_over_ch=float(g.std(0).mean()),
                         gate_std_max_over_ch=float(g.std(0).max()))
    return rows


def main():
    os.makedirs(RUNS, exist_ok=True)
    from phase3.code.rgbt_trainer import RGBTDetectionTrainer
    import inspect
    assert "pad=0.0" in inspect.getsource(RGBTDetectionTrainer.build_dataset), "fixed validator missing -- STOP"
    device = rsr.get_device()
    results = {}
    for name, flags in VARIANTS.items():
        t0 = time.time()
        run_dir = train(name, flags, device)
        print(f"[{name}] training done in {time.time() - t0:.0f}s")
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(os.path.join(run_dir, "results.csv")))]
        last_row = rows[-1]
        reval = rsr.revalidate_checkpoint(os.path.join(run_dir, "weights", "last.pt"), device)
        a2 = a2_table(os.path.join(run_dir, "weights", "last.pt"), flags)
        results[name] = {"epochs": len(rows), "inloop_epoch5": {k: float(last_row[k]) for k in (
            "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)")},
            "reval_last": reval, "a2": a2}
        json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
        print(f"[{name}] {json.dumps(results[name], default=float)}")

    off, on = results["b6_off"], results["b6_on"]
    L = ["# B6 pilot — cbam_rgvqf, seed 0, 5 epochs, B1+B2+B3+B5 ON\n\n",
         "| variant | epochs | mAP50 (last.pt reval) | mAP50-95 (last.pt reval) | P@.25 | R@.25 |\n|---|---|---|---|---|---|\n"]
    for n, r in results.items():
        rv = r["reval_last"]
        L.append(f"| {n} | {r['epochs']} | {rv['mAP50']:.4f} | {rv['mAP50_95']:.4f} | {rv['P']:.3f} | {rv['R']:.3f} |\n")
    L.append("\n| scale | <Z> std OFF | <Z> std ON | gate std OFF (mean / max ch) | gate std ON (mean / max ch) | gate-std ratio ON/OFF |\n|---|---|---|---|---|---|\n")
    ratios = []
    for lvl in ("p3", "p4", "p5"):
        a, b = off["a2"][lvl], on["a2"][lvl]
        ratio = b["gate_std_mean_over_ch"] / max(a["gate_std_mean_over_ch"], 1e-12)
        ratios.append(ratio)
        L.append(f"| {lvl} | {a['mean_qubit_std']:.4f} | {b['mean_qubit_std']:.4f} | {a['gate_std_mean_over_ch']:.5f} / {a['gate_std_max_over_ch']:.5f} | "
                 f"{b['gate_std_mean_over_ch']:.5f} / {b['gate_std_max_over_ch']:.5f} | {ratio:.1f}x |\n")
    d = on["reval_last"]["mAP50_95"] - off["reval_last"]["mAP50_95"]
    d50 = on["reval_last"]["mAP50"] - off["reval_last"]["mAP50"]
    crit1 = all(r >= 10 for r in ratios)
    crit2 = d >= 0
    L.append(f"\n- Criterion 1 (gate std >= 10x at every scale): **{'PASS' if crit1 else 'FAIL'}** (ratios {[round(r, 1) for r in ratios]})\n")
    L.append(f"- Criterion 2 (B6 ON mAP50-95 not lower): **{'PASS' if crit2 else 'FAIL'}** (Δ mAP50-95 = {d:+.4f}, Δ mAP50 = {d50:+.4f}; "
             "single seed — seed-to-seed SD of paired diffs at 50 ep was ~0.018)\n")
    open(os.path.join(OUT, "summary.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
