#!/usr/bin/env python3
"""
Canary analysis (read-only): per-epoch val mAP + train losses, A2 at epochs 10/25/50, composite-loss terms per
epoch (TRAIN batches only, parsed from canary.log), measured sec/epoch, and costed EPOCHS options.
Writes final_runs/canary_analysis.md.
    .venv/bin/python -I diagnostics/presubmission/canary_analysis.py
"""
import csv
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "final_runs")
RUN = os.path.join(FR, "runs", "abl_complete_s0")

rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(os.path.join(RUN, "results.csv")))]
res = json.load(open(os.path.join(RUN, "result.json")))
a2 = json.load(open(os.path.join(RUN, "a2.json")))
summ = next(r for r in csv.DictReader(open(os.path.join(FR, "summary.csv"))) if r["run"] == "abl_complete_s0")

# ---- composite loss terms, train batches only
txt = open(os.path.join(FR, "canary.log"), errors="replace").read().replace("\r", "\n")
pat = re.compile(r"\[composite loss\] det=([-\d.]+) l_rel=([-\d.]+) \(x[\d.]+\) l_q=([-\d.]+)")
ep_pat = re.compile(r"^\s*(?:\x1b\[K)?\s*(\d+)/50\s")
terms, ctx_epoch, ctx = {}, None, None
n_train = n_val = 0
for line in txt.split("\n"):
    m_ep = ep_pat.search(line.replace("\x1b[K", ""))
    if m_ep:
        ctx_epoch, ctx = int(m_ep.group(1)), "train"
    elif "Box(P" in line or "Class     Images" in line:
        ctx = "val"
    for m in pat.finditer(line):
        if ctx == "train" and ctx_epoch:
            terms.setdefault(ctx_epoch, []).append(tuple(float(x) for x in m.groups()))
            n_train += 1
        else:
            n_val += 1

# ---- per-epoch table
sec = []
prev = 0.0
for r in rows:
    t = float(r["time"])
    sec.append(t - prev)
    prev = t
L = ["# Canary analysis — abl_complete, seed 0, 50 epochs, all fixes ON, k=2\n\n",
     f"Reported (last.pt, fixed validator): mAP50 {res['reval_last']['mAP50']:.4f} · mAP50-95 {res['reval_last']['mAP50_95']:.4f} · "
     f"P@.25 {res['reval_last']['P']:.3f} · R@.25 {res['reval_last']['R']:.3f}\n\n",
     "## 1. Per epoch (val = in-loop, EMA weights, fixed validator; train losses = epoch means as logged by Ultralytics)\n\n",
     "| epoch | val mAP50 | val mAP50-95 | train box | train cls | train dfl | val box | val cls | val dfl | mean l_rel | mean l_q | s/epoch |\n",
     "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
for i, r in enumerate(rows):
    e = int(float(r["epoch"]))
    t = terms.get(e, [])
    lr_, lq_ = (st.mean(x[1] for x in t), st.mean(x[2] for x in t)) if t else (float("nan"), float("nan"))
    L.append(f"| {e} | {float(r['metrics/mAP50(B)']):.4f} | {float(r['metrics/mAP50-95(B)']):.4f} | {float(r['train/box_loss']):.4f} | "
             f"{float(r['train/cls_loss']):.4f} | {float(r['train/dfl_loss']):.4f} | {float(r['val/box_loss']):.4f} | "
             f"{float(r['val/cls_loss']):.4f} | {float(r['val/dfl_loss']):.4f} | {lr_:+.4f} | {lq_:+.4f} | {sec[i]:.0f} |\n")
m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
pk = max(range(len(m)), key=lambda i: m[i])
L.append(f"\nPeak in-loop mAP50-95 {m[pk]:.4f} at epoch {pk + 1}; epoch 50 {m[-1]:.4f} (Δ {m[-1] - m[pk]:+.4f}). "
         f"Composite prints parsed: {n_train} train (expected {1740 * 50}), {n_val} val/other (excluded).\n")

# ---- A2
L.append("\n## 2. A2 at epochs 10 / 25 / 50 (100 random val images; eval batch 1, train-mode shuffled batches of 4)\n\n"
         "| epoch | scale | <Z> std (eval) | gate std mean / max (eval) | |x|>0.9π eval | |x|>0.9π train | mean |γ| | effective gain |γ|/k |\n"
         "|---|---|---|---|---|---|---|---|\n")
for ep in ("epoch10", "epoch25", "epoch50"):
    for lvl in ("p3", "p4", "p5"):
        e, t = a2[ep][lvl]["eval_b1"], a2[ep][lvl]["train_b4"]
        g = sum(abs(v) for v in e["bn_gamma"]) / len(e["bn_gamma"])
        L.append(f"| {ep[5:]} | {lvl} | {e['mean_qubit_std']:.4f} | {e['gate_std_mean']:.5f} / {e['gate_std_max']:.5f} | "
                 f"{e['x_sat_frac']:.3f} | {t['x_sat_frac']:.3f} | {g:.3f} | {e['effective_gain']:.3f} |\n")

# ---- timing + costing
train_s = [s for s in sec]
med = st.median(train_s)
overhead_s = 15 * 60  # reval + Ultralytics final_eval + A2 (RGVQF configs); estimate from canary wall vs epoch sum
canary_total = float(summ["train_wall_s"])
L.append(f"\n## 4. Timing\n\nMeasured sec/epoch (train + in-loop val, MPS): median {med:.0f} s, mean {st.mean(train_s):.0f} s, "
         f"min {min(train_s):.0f}, max {max(train_s):.0f}. Epoch-1 includes setup. Canary train wall {canary_total / 3600:.1f} h.\n")
# per-config sec/epoch: scale the old grid medians (phase2_runs.csv) by canary / old-complete ratio
old = {}
for r in csv.DictReader(open(os.path.join(HERE, "phase2_runs.csv"))):
    old.setdefault(r["run"], []).append(float(r["sec_per_epoch"]))
old_med = {k: st.median(v) for k, v in old.items()}
ratio = med / old_med["abl_complete"]
cfg_sec = {k: v * ratio for k, v in old_med.items()}
cfg_sec["abl_cbam_rgvqf_classical_ctrl"] = cfg_sec["abl_cbam_no_vqc"] * 1.05  # classical gate, no QNode: ~cbam_add + small
base_sec = cfg_sec["abl_no_cbam_add"] * 0.55  # single stream: one backbone, no fusion -- UNVERIFIED estimate
fusion11 = sum(cfg_sec[k] for k in cfg_sec)
L.append(f"\nPer-config sec/epoch estimate = old step-5 median × {ratio:.3f} (canary / old abl_complete); classical control ≈ "
         f"cbam_add × 1.05; single-stream baseline ≈ 0.55 × no_cbam_add (**UNVERIFIED**, measure in the first baseline epoch). "
         f"Σ over 11 fusion configs = {fusion11:.0f} s/epoch; baseline {base_sec:.0f} s/epoch.\n\n")
L.append("| EPOCHS | KEEP_EPOCHS | (a) 11 fusion + 2 baselines, seed 0 | (b) + seeds 1–4 of complete & cbam_add (8 runs) | (c) + seeds 1–7 (14 runs) |\n|---|---|---|---|---|\n")
for E, keep in ((10, (5, 10)), (12, (6, 12)), (15, (5, 10, 15))):
    a = E * (fusion11 + 2 * base_sec) + 13 * overhead_s
    pair = E * (cfg_sec["abl_complete"] + cfg_sec["abl_cbam_no_vqc"]) + 2 * overhead_s
    b, c = a + 4 * pair, a + 7 * pair
    L.append(f"| {E} | {keep} | {a / 3600:.0f} h ({a / 86400:.1f} d) | {b / 3600:.0f} h ({b / 86400:.1f} d) | {c / 3600:.0f} h ({c / 86400:.1f} d) |\n")
L.append(f"\nOverhead per run assumed {overhead_s / 60:.0f} min (re-validation + Ultralytics final_eval + A2). Sequential, single M1 Pro.\n")
open(os.path.join(FR, "canary_analysis.md"), "w").write("".join(L))
print("".join(L))
