#!/usr/bin/env python3
"""
Campaign (EPOCHS=12) results under the pre-registered reporting rule. READ-ONLY on campaign_e12/runs and
summary.csv; writes campaign_e12/RESULTS.md only. Thermal seeds 1-7 (running separately) are excluded until §6.
    .venv/bin/python -I diagnostics/presubmission/campaign_results.py
"""
import csv
import glob
import json
import math
import os
import statistics as st

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "campaign_e12")
FREEZE = json.load(open(os.path.join(HERE, "CODE_FREEZE.json")))
rows = {r["run"]: r for r in csv.DictReader(open(os.path.join(C, "summary.csv")))}

SEED0 = [("base_yolo11n_thermal_s0", "Thermal-only YOLO11n (baseline)"), ("base_yolo11n_rgb_s0", "RGB-only YOLO11n (baseline)"),
         ("abl_no_cbam_add_s0", "no attention + add"), ("abl_cbam_no_vqc_s0", "CBAM + add (cbam_add)"),
         ("abl_cbam_vqc_proposed_s0", "CBAM + quantum-inspired (classical)"), ("abl_cbam_rgvqf_proposed_s0", "CBAM + RGVQF"),
         ("abl_cbam_rgvqf_no_entangle_s0", "CBAM + RGVQF, no entanglement"), ("abl_cbam_rgvqf_rem_s0", "CBAM + REM + RGVQF"),
         ("abl_cbam_rgvqf_classical_ctrl_s0", "CBAM + RGVQF encoder, classical circuit (control)"),
         ("abl_cmaam_add_s0", "CMAAM + add"), ("abl_cmaam_rgvqf_rem_s0", "CMAAM + REM + RGVQF"),
         ("abl_mqfp_single_scale_s0", "CMAAM + REM + RGVQF, P5 only (single-scale)"),
         ("abl_complete_s0", "Complete (CMAAM + REM + RGVQF + composite loss)")]
CAMPAIGN = [r for r, _ in SEED0] + [f"abl_{c}_s{s}" for s in range(1, 8) for c in ("complete", "cbam_no_vqc")]
missing = [r for r in CAMPAIGN if r not in rows]
assert not missing, f"missing campaign rows: {missing}"
f = lambda r, k: float(rows[r][k])  # noqa: E731

L = ["# Q-WildFusion — pre-submission campaign results (EPOCHS = 12)\n\n",
     f"Code freeze `{FREEZE['code_hash'][:12]}` (previous `{FREEZE['previous_hash'][:12]}`; reason: {FREEZE['reason']}). "
     "All runs: dual-stream YOLO11n on LLVIP (custom 6,957 / 1,740 split), fixed validator (val 768×960, no padding), "
     "all approved fixes ON (B1 LeakyReLU channel attention, B2 aux-loss ×B, B3 REM only at fused levels, B5 seeded data order, "
     "B6 learned encoder with k = 2), 12 epochs, no early stopping.\n\n",
     "**Pre-registered reporting rule.** Primary: " + FREEZE["reporting_rule"]["primary"] + ". Secondary: " +
     FREEZE["reporting_rule"]["secondary"] + ". " + FREEZE["reporting_rule"]["collapse"].capitalize() + ".\n\n"]

# ---------------- 1
L.append("## 1. Seed-0 grid (11 fusion configs + 2 baselines)\n\n"
         "| run | config | mAP50 | **mAP50-95** (primary) | mean last 3 (secondary) | P@0.25 | R@0.25 | params used | quantum params | collapse_flag |\n"
         "|---|---|---|---|---|---|---|---|---|---|\n")
for r, label in SEED0:
    L.append(f"| {label} | `{rows[r]['config']}` | {f(r, 'mAP50'):.4f} | **{f(r, 'mAP50_95'):.4f}** | {f(r, 'mean_last3_mAP50_95'):.4f} | "
             f"{f(r, 'P_at_0_25'):.3f} | {f(r, 'R_at_0_25'):.3f} | {int(rows[r]['params_used']):,} | {rows[r]['params_quantum']} | "
             f"{rows[r]['collapse_flag']} |\n")
L.append("\nP and R are read at confidence 0.25 from the PR curve; mAP is threshold-free. Params used exclude the unused "
         "stock backbone copy kept in the checkpoint (1,365,472).\n")


# ---------------- 2
def paired(metric, pa="abl_complete", pb="abl_cbam_no_vqc"):
    a = [f(f"{pa}_s{s}", metric) for s in range(8)]
    b = [f(f"{pb}_s{s}", metric) for s in range(8)]
    d = [x - y for x, y in zip(a, b)]
    md, sd = st.mean(d), st.stdev(d)
    half = stats.t.ppf(0.975, 7) * sd / math.sqrt(8)
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    return a, b, d, dict(mean_a=st.mean(a), sd_a=st.stdev(a), mean_b=st.mean(b), sd_b=st.stdev(b), md=md, sd=sd,
                         lo=md - half, hi=md + half, t=t.statistic, p=t.pvalue, w=w.pvalue, dz=md / sd,
                         pos=sum(x > 0 for x in d))


L.append("\n## 2. Paired test: complete vs cbam_add, seeds 0–7 (n = 8)\n\n"
         "Same seed ⇒ same data order (B5) and identical COCO-pretrained backbone/neck weights; randomly initialised layers are not shared (different modules consume the RNG differently). Two-sided tests.\n")
summary2 = {}
for metric, label in (("mAP50_95", "mAP50-95 (primary)"), ("mAP50", "mAP50 (secondary)"),
                      ("mean_last3_mAP50_95", "mean of in-loop mAP50-95, epochs 10–12 (secondary)")):
    a, b, d, s = paired(metric)
    summary2[metric] = s
    L.append(f"\n### {label}\n\n| seed | complete | cbam_add | difference |\n|---|---|---|---|\n")
    for i in range(8):
        L.append(f"| {i} | {a[i]:.4f} | {b[i]:.4f} | {d[i]:+.4f} |\n")
    L.append(f"\n- complete {s['mean_a']:.4f} ± {s['sd_a']:.4f} · cbam_add {s['mean_b']:.4f} ± {s['sd_b']:.4f} (mean ± SD)\n"
             f"- mean paired difference **{s['md']:+.4f}** (SD {s['sd']:.4f}), 95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]\n"
             f"- paired t({7}) = {s['t']:.3f}, **p = {s['p']:.4f}** · Wilcoxon signed-rank **p = {s['w']:.4f}** · "
             f"Cohen's d_z = {s['dz']:+.3f} · {s['pos']}/8 seeds favour complete\n")
p = summary2["mAP50_95"]
sig_t, sig_w = p["p"] < 0.05, p["w"] < 0.05
direction = "lower" if p["md"] < 0 else "higher"
if sig_t and sig_w:
    verdict = f"complete is significantly {direction} than cbam_add (both tests p < 0.05)."
elif sig_t or sig_w:
    verdict = (f"Borderline: the two pre-specified tests disagree at α = 0.05 (paired t p = {p['p']:.3f}, "
               f"Wilcoxon p = {p['w']:.3f}). The direction is consistent: complete is {direction} than cbam_add in "
               f"{8 - p['pos'] if p['md'] < 0 else p['pos']}/8 seeds.")
else:
    verdict = "No significant difference at α = 0.05 by either test."
L.append(f"\n**Verdict (primary).** complete − cbam_add = {p['md']:+.4f} mAP50-95, 95% CI [{p['lo']:+.4f}, {p['hi']:+.4f}]. "
         f"{verdict} The CI upper bound ({p['hi']:+.4f}) rules out a gain for complete larger than that; no analysis in this "
         "section shows complete outperforming cbam_add. Secondary metrics (mAP50, mean last 3) are not significant.\n")
L.append("\nNote: complete seed 2 dipped over epochs 10–11 (0.576 → 0.534 → 0.443 → 0.575 at epoch 12). No single-epoch drop "
         "exceeded 0.10, so no collapse copy or flag was triggered; per the rule it is reported as-is. It lowers that run's "
         "secondary metric (mean last 3 = 0.517), not the primary.\n")

# ---------------- 3
coll_files = sorted(glob.glob(os.path.join(C, "runs", "*", "weights", "collapse_*.pt")))
coll_logs = sorted(glob.glob(os.path.join(C, "runs", "*", "collapse_log.txt")))
flags = [r for r in CAMPAIGN if rows[r]["collapse_flag"] == "True"]
L.append("\n## 3. Collapse summary (27 campaign runs)\n\n"
         f"- collapse_epoch*.pt files: **{len(coll_files)}** {[os.path.relpath(x, C) for x in coll_files] or ''}\n"
         f"- collapse_log.txt files: **{len(coll_logs)}**\n- collapse_flag = True: **{len(flags)}** {flags or ''}\n"
         "- Near-miss: complete s2, epochs 10–11 (see §2 note). Largest single-epoch drop in any run is listed below.\n\n")
drops = []
for r in CAMPAIGN:
    rr = [float(x["metrics/mAP50-95(B)"]) for x in csv.DictReader(open(os.path.join(C, "runs", r, "results.csv")))]
    dmax = max((rr[i - 1] - rr[i], i + 1) for i in range(1, len(rr)))
    drops.append((dmax[0], r, dmax[1]))
drops.sort(reverse=True)
L.append("| run | largest single-epoch drop in in-loop mAP50-95 | at epoch |\n|---|---|---|\n")
for dv, r, e in drops[:5]:
    L.append(f"| {r} | {dv:.4f} | {e} |\n")

# ---------------- 4
L.append("\n## 4. A2 at epoch 12 (RGVQF-type configs; 100 random val images, eval mode, batch 1)\n\n"
         "| run | scale | ⟨Z⟩ std | gate std mean / max | saturation (|x|>0.9π) | train-mode saturation | effective gain |γ|/k |\n"
         "|---|---|---|---|---|---|---|\n")
a2_runs = [r for r, _ in SEED0 if os.path.isfile(os.path.join(C, "runs", r, "a2.json"))]
for r in a2_runs:
    a2 = json.load(open(os.path.join(C, "runs", r, "a2.json")))["epoch12"]
    for lvl in ("p3", "p4", "p5"):
        if lvl not in a2:
            continue
        e, t = a2[lvl]["eval_b1"], a2[lvl]["train_b4"]
        L.append(f"| {r.replace('_s0', '')} | {lvl} | {e['mean_qubit_std']:.3f} | {e['gate_std_mean']:.4f} / {e['gate_std_max']:.4f} | "
                 f"{e['x_sat_frac']:.3f} | {t['x_sat_frac']:.3f} | {e['effective_gain']:.3f} |\n")
cs = [json.load(open(os.path.join(C, "runs", f"abl_complete_s{s}", "a2.json")))["epoch12"] for s in range(8)]
L.append("\ncomplete, seeds 0–7 at epoch 12 (mean [min–max] over seeds):\n\n| scale | ⟨Z⟩ std | gate std mean | saturation eval | effective gain |\n|---|---|---|---|---|\n")
for lvl in ("p3", "p4", "p5"):
    def agg(k):
        v = [c[lvl]["eval_b1"][k] for c in cs]
        return f"{st.mean(v):.3f} [{min(v):.3f}–{max(v):.3f}]"
    L.append(f"| {lvl} | {agg('mean_qubit_std')} | {agg('gate_std_mean')} | {agg('x_sat_frac')} | {agg('effective_gain')} |\n")
L.append("\nFor the classical control, ⟨Z⟩ is the output of tanh(Linear(8,8)(x)) in place of the circuit.\n")

# ---------------- 5
a_sd = st.mean([summary2["mAP50_95"]["sd_a"], summary2["mAP50_95"]["sd_b"]])
noise = math.sqrt(2) * a_sd
L.append(f"\n## 5. Single-factor comparisons (seed 0 only)\n\n**Noise reminder.** Seed-to-seed SD of mAP50-95 within a config is "
         f"~{a_sd:.4f} (complete / cbam_add, n = 8), so a single-seed difference between two configs has SD ≈ {noise:.4f}. "
         "**Single-seed gaps under ~0.02 are within noise** and must not be reported as effects.\n\n"
         "| factor | A | B | Δ mAP50-95 (A − B) | Δ mean last 3 | Δ mAP50 | within noise (<0.02)? |\n|---|---|---|---|---|---|---|\n")
COMPS = [("entanglement on vs off", "abl_cbam_rgvqf_proposed_s0", "abl_cbam_rgvqf_no_entangle_s0"),
         ("REM on vs off (CBAM + RGVQF)", "abl_cbam_rgvqf_rem_s0", "abl_cbam_rgvqf_proposed_s0"),
         ("CMAAM vs CBAM (add fusion)", "abl_cmaam_add_s0", "abl_cbam_no_vqc_s0"),
         ("CMAAM vs CBAM (REM + RGVQF)", "abl_cmaam_rgvqf_rem_s0", "abl_cbam_rgvqf_rem_s0"),
         ("multi-scale vs single-scale (P5 only)", "abl_cmaam_rgvqf_rem_s0", "abl_mqfp_single_scale_s0"),
         ("composite loss on vs off", "abl_complete_s0", "abl_cmaam_rgvqf_rem_s0"),
         ("RGVQF (quantum) vs classical control", "abl_cbam_rgvqf_proposed_s0", "abl_cbam_rgvqf_classical_ctrl_s0"),
         ("RGVQF vs quantum-inspired (classical)", "abl_cbam_rgvqf_proposed_s0", "abl_cbam_vqc_proposed_s0"),
         ("RGVQF vs add", "abl_cbam_rgvqf_proposed_s0", "abl_cbam_no_vqc_s0"),
         ("quantum-inspired vs add", "abl_cbam_vqc_proposed_s0", "abl_cbam_no_vqc_s0"),
         ("CBAM vs no attention (add)", "abl_cbam_no_vqc_s0", "abl_no_cbam_add_s0"),
         ("best fusion (cbam_add) vs thermal-only", "abl_cbam_no_vqc_s0", "base_yolo11n_thermal_s0"),
         ("complete vs thermal-only", "abl_complete_s0", "base_yolo11n_thermal_s0")]
for lab, A, B in COMPS:
    d1, d2, d3 = (f(A, k) - f(B, k) for k in ("mAP50_95", "mean_last3_mAP50_95", "mAP50"))
    L.append(f"| {lab} | `{rows[A]['config']}` | `{rows[B]['config']}` | {d1:+.4f} | {d2:+.4f} | {d3:+.4f} | "
             f"{'yes' if abs(d1) < 0.02 else '**no**'} |\n")
L.append("\nIndividual seed-0 values are in §1. Spread of the 11 fusion configs at seed 0: mAP50-95 "
         f"{min(f(r, 'mAP50_95') for r, _ in SEED0[2:]):.4f}–{max(f(r, 'mAP50_95') for r, _ in SEED0[2:]):.4f}.\n")

# ---------------- 6
THERM = [f"base_yolo11n_thermal_s{s}" for s in range(8)]
miss = [r for r in THERM if r not in rows]
assert not miss, f"missing thermal rows: {miss}"
tflags = [r for r in THERM if rows[r]["collapse_flag"] == "True"]
tfiles = sorted(glob.glob(os.path.join(C, "runs", "base_yolo11n_thermal_s*", "weights", "collapse_*.pt")))
L.append("\n## 6. Fusion vs thermal-only, seeds 0–7 (n = 8)\n\n"
         "Thermal-only = stock single-stream YOLO11n (COCO-pretrained, 2,590,035 params) on the IR images only, standard "
         "Ultralytics trainer with the same fixed validator (768×960) and seeded data order (B5), HSV off and no Albumentations "
         "(as the fusion IR stream), otherwise identical settings and 12 epochs. Same seed ⇒ same data order as the fusion runs; "
         "the initialisation is not shared (different architecture), so pairing is by seed and data order. All 8 thermal runs: "
         f"code hash {FREEZE['code_hash'][:12]}, 12 epochs, collapse_flag = True: {len(tflags)}, collapse_epoch*.pt files: {len(tfiles)}.\n")
for pa, la in (("abl_complete", "complete"), ("abl_cbam_no_vqc", "cbam_add")):
    L.append(f"\n### {la} vs thermal-only\n")
    for metric, label in (("mAP50_95", "mAP50-95 (primary)"), ("mAP50", "mAP50 (secondary)"),
                          ("mean_last3_mAP50_95", "mean of in-loop mAP50-95, epochs 10–12 (secondary)")):
        a, b, d, q = paired(metric, pa, "base_yolo11n_thermal")
        L.append(f"\n**{label}**\n\n| seed | {la} | thermal-only | difference |\n|---|---|---|---|\n")
        for i in range(8):
            L.append(f"| {i} | {a[i]:.4f} | {b[i]:.4f} | {d[i]:+.4f} |\n")
        L.append(f"\n- {la} {q['mean_a']:.4f} ± {q['sd_a']:.4f} · thermal-only {q['mean_b']:.4f} ± {q['sd_b']:.4f} (mean ± SD)\n"
                 f"- mean paired difference **{q['md']:+.4f}** (SD {q['sd']:.4f}), 95% CI [{q['lo']:+.4f}, {q['hi']:+.4f}]\n"
                 f"- paired t(7) = {q['t']:.3f}, **p = {q['p']:.4f}** · Wilcoxon signed-rank **p = {q['w']:.4f}** · "
                 f"Cohen's d_z = {q['dz']:+.3f} · {q['pos']}/8 seeds favour {la}\n")
        if metric == "mAP50_95":
            st_t, st_w = q["p"] < 0.05, q["w"] < 0.05
            dirn = "lower" if q["md"] < 0 else "higher"
            v = (f"{la} is significantly {dirn} than thermal-only (both tests p < 0.05)." if st_t and st_w else
                 f"Borderline: the tests disagree at α = 0.05 (t p = {q['p']:.3f}, Wilcoxon p = {q['w']:.3f})." if st_t or st_w else
                 "No significant difference at α = 0.05 by either test.")
            L.append(f"- *Primary verdict:* {v} The data are compatible with a {la} advantage over thermal-only of at most "
                     f"{q['hi']:+.4f} mAP50-95 (upper 95% CI bound).\n")

open(os.path.join(C, "RESULTS.md"), "w").write("".join(L))
print("".join(L))
