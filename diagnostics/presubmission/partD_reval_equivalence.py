#!/usr/bin/env python3
"""
Part D check: run_final.revalidate() (explicit grid settings, symlinked dataset) must reproduce the Phase 2
number from run_step5_retrain.revalidate_checkpoint (reval_fixed_geometry.csv) on an existing checkpoint.
    caffeinate -i -s .venv/bin/python -u diagnostics/presubmission/partD_reval_equivalence.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
import run_final as rf  # noqa: E402

W = os.path.join(rf.ROOT, "diagnostics/step5_retrain/runs/abl_cbam_no_vqc/weights/best.pt")
ref = next(r for r in csv.DictReader(open(os.path.join(HERE, "reval_fixed_geometry.csv")))
           if r["run"] == "abl_cbam_no_vqc" and r["seed"] == "0" and r["ckpt"] == "best.pt")
new = rf.revalidate("cbam_add", W, "mps", os.path.join(HERE, "smoke_runs"))
pairs = [("mAP50", ref["mAP50_new"], new["mAP50"]), ("mAP50-95", ref["mAP50-95_new"], new["mAP50_95"]),
         ("P", ref["P"], new["P"]), ("R", ref["R"], new["R"])]
for k, a, b in pairs:
    print(f"{k:9s} phase2={float(a):.6f} driver={b:.6f} |diff|={abs(float(a) - b):.2e}")
ok = all(abs(float(a) - b) < 1e-6 for _, a, b in pairs)
print("REVAL EQUIVALENCE:", "PASS" if ok else "FAIL")
