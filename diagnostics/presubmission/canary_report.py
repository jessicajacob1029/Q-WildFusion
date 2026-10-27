#!/usr/bin/env python3
"""
Canary report for final_runs/runs/abl_complete_s0 (read-only):
per-epoch mAP50 / mAP50-95 curve, peak epoch, last-epoch value, the fixed-validator re-validation of last.pt,
and the A2 table at epochs 10/25/50 (eval b1 and train-mode shuffled b4): <Z> std, gate std, |x|>0.9pi saturation,
encoder BN gamma/beta, running stats, effective gain = mean|gamma| / k.
    .venv/bin/python -I diagnostics/presubmission/canary_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "final_runs", "runs", "abl_complete_s0")

res = json.load(open(os.path.join(RUN, "result.json")))
man = json.load(open(os.path.join(RUN, "manifest.json")))
a2 = json.load(open(os.path.join(RUN, "a2.json")))
pe = res["per_epoch"]
peak = max(pe, key=lambda r: r[2])
L = [f"# Canary — abl_complete, seed 0, {len(pe)} epochs (patience 50), all fixes ON, k = {man.get('encoder_gain_k')}\n\n",
     f"Code hash `{man.get('code_hash')}` · params {man.get('param_report')}\n\n",
     f"- **Reported result (last.pt, fixed validator):** mAP50 {res['reval_last']['mAP50']:.4f} · mAP50-95 "
     f"{res['reval_last']['mAP50_95']:.4f} · P@.25 {res['reval_last']['P']:.3f} · R@.25 {res['reval_last']['R']:.3f}\n",
     f"- In-loop peak mAP50-95 {peak[2]:.4f} at epoch {peak[0]}; last epoch ({pe[-1][0]}) {pe[-1][2]:.4f} "
     f"(last − peak {pe[-1][2] - peak[2]:+.4f})\n\n",
     "## Per-epoch (in-loop, EMA, fixed validator)\n\n| epoch | mAP50 | mAP50-95 |\n|---|---|---|\n"]
L += [f"| {e} | {m50:.4f} | {m5095:.4f} |\n" for e, m50, m5095 in pe]
L.append("\n## A2 at epochs 10 / 25 / 50 (100 random val images; eval batch 1 · train-mode shuffled batches of 4)\n\n"
         "| epoch | scale | <Z> std eval / train | gate std mean (max ch) eval | gate std mean train | |x|>0.9π eval / train "
         "| x std eval | BN gamma mean|·| | effective gain |γ|/k | BN running_var mean |\n|---|---|---|---|---|---|---|---|---|---|\n")
for ep in ("epoch10", "epoch25", "epoch50"):
    if ep not in a2:
        L.append(f"| {ep} | missing | | | | | | | | |\n")
        continue
    for lvl in ("p3", "p4", "p5"):
        e, t = a2[ep][lvl]["eval_b1"], a2[ep][lvl]["train_b4"]
        g = e["bn_gamma"]
        L.append(f"| {ep[5:]} | {lvl} | {e['mean_qubit_std']:.4f} / {t['mean_qubit_std']:.4f} | {e['gate_std_mean']:.5f} ({e['gate_std_max']:.5f}) | "
                 f"{t['gate_std_mean']:.5f} | {e['x_sat_frac']:.3f} / {t['x_sat_frac']:.3f} | {e['x_std_mean']:.3f} | "
                 f"{sum(abs(v) for v in g) / len(g):.3f} | {e['effective_gain']:.3f} | {sum(e['bn_running_var']) / len(e['bn_running_var']):.4f} |\n")
L.append("\nPer-dim BN gamma / beta (eval record):\n\n")
for ep in ("epoch10", "epoch25", "epoch50"):
    for lvl in ("p3", "p4", "p5"):
        if ep in a2:
            e = a2[ep][lvl]["eval_b1"]
            L.append(f"- {ep} {lvl}: gamma {e['bn_gamma']} · beta {e['bn_beta']}\n")
open(os.path.join(HERE, "final_runs", "canary_report.md"), "w").write("".join(L))
print("".join(L))
