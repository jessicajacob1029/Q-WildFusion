#!/usr/bin/env python3
"""
Flags-OFF reproducibility: build models with the HEAD code (exported to
preflight/_oldcode via `git show`) and with the current code (all new flags at
their default False), same seed and construction order, and compare
state_dict bytes + the loss vector on the same train batch.

    .venv/bin/python -u diagnostics/presubmission/preflight/partB_repro.py --code old > old.json
    .venv/bin/python -u diagnostics/presubmission/preflight/partB_repro.py --code new > new.json
    .venv/bin/python -u diagnostics/presubmission/preflight/partB_repro.py --compare
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ["cbam_add", "cmaam_add", "cbam_rgvqf", "cmaam_rgvqf_rem_single_scale", "complete"]


def run(code):
    sys.path.insert(0, os.path.join(HERE, "_oldcode") if code == "old" else ROOT)
    sys.path.insert(1, ROOT)  # run_* modules etc. never needed, but keep repo importable
    os.chdir(ROOT)
    import torch
    from ultralytics import YOLO

    import phase3.code.rgbt_trainer as rt
    from phase3.code.train_ablation_study import ABLATIONS, make_trainer
    assert ("_oldcode" in rt.__file__) == (code == "old"), rt.__file__
    out = {"code_file": rt.__file__}
    for key in CONFIGS:
        base = YOLO("yolo11n.pt")
        tr = make_trainer(ABLATIONS[key])(overrides=dict(
            data="phase3/dataset_aligned/data.yaml", imgsz=960, batch=4, device="cpu", workers=0, rect=True,
            seed=0, deterministic=True, model="yolo11n.pt", project=os.path.join(HERE, "_scratch"),
            name=f"repro_{code}_{key}", exist_ok=True, plots=False, epochs=1, box=10.0, cls=0.3, dfl=1.5,
            mosaic=0.0, fliplr=0.0, translate=0.0, scale=0.0))
        m = tr.get_model(cfg=base.model.yaml, weights=base.model, verbose=False)
        tr.model = m
        tr.set_model_attributes()
        h = hashlib.sha256()
        for k, v in m.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        dl = tr.get_dataloader(tr.data["train"], batch_size=4, rank=-1, mode="train")
        torch.manual_seed(123)
        b = tr.preprocess_batch(next(iter(dl)))
        m.train()
        loss, items = m(b)
        out[key] = {"state_sha256": h.hexdigest(), "n_tensors": len(m.state_dict()),
                    "loss": [float(x) for x in loss], "loss_items": [float(x) for x in items]}
    print("JSON" + json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["old", "new"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        o, n = (json.load(open(os.path.join(HERE, f"partB_repro_{c}.json"))) for c in ("old", "new"))
        ok = True
        for key in CONFIGS:
            same_w = o[key]["state_sha256"] == n[key]["state_sha256"]
            same_l = o[key]["loss"] == n[key]["loss"]
            ok &= same_w and same_l
            print(f"{key:30s} weights identical={same_w} ({o[key]['n_tensors']} vs {n[key]['n_tensors']} tensors) "
                  f"loss identical={same_l} old={[round(x, 6) for x in o[key]['loss']]} new={[round(x, 6) for x in n[key]['loss']]}")
        print("FLAGS-OFF REPRODUCES HEAD:", "PASS" if ok else "FAIL")
    else:
        run(a.code)
