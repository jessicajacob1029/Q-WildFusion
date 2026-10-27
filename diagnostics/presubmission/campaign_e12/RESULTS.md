# Q-WildFusion — pre-submission campaign results (EPOCHS = 12)

Code freeze `2dc553937017` (previous `a8c156668c09`; reason: training length set from canary val-loss divergence; collapse logging added). All runs: dual-stream YOLO11n on LLVIP (custom 6,957 / 1,740 split), fixed validator (val 768×960, no padding), all approved fixes ON (B1 LeakyReLU channel attention, B2 aux-loss ×B, B3 REM only at fused levels, B5 seeded data order, B6 learned encoder with k = 2), 12 epochs, no early stopping.

**Pre-registered reporting rule.** Primary: last.pt at the final epoch (12), re-validated with the fixed validator (val pad 0, 768x960). Secondary: mean of the in-loop mAP50-95 over the last 3 epochs (10-12). A collapsed run is reported as-is with its collapse_flag; no run is ever re-run or replaced because of its result.

## 1. Seed-0 grid (11 fusion configs + 2 baselines)

| run | config | mAP50 | **mAP50-95** (primary) | mean last 3 (secondary) | P@0.25 | R@0.25 | params used | quantum params | collapse_flag |
|---|---|---|---|---|---|---|---|---|---|
| Thermal-only YOLO11n (baseline) | `baseline_thermal` | 0.9064 | **0.5787** | 0.5823 | 0.567 | 0.932 | 2,590,035 | 0 | False |
| RGB-only YOLO11n (baseline) | `baseline_rgb` | 0.7562 | **0.3972** | 0.3983 | 0.807 | 0.715 | 2,590,035 | 0 | False |
| no attention + add | `no_cbam_add` | 0.9080 | **0.5658** | 0.5687 | 0.583 | 0.931 | 3,955,507 | 0 | False |
| CBAM + add (cbam_add) | `cbam_add` | 0.9197 | **0.5781** | 0.5791 | 0.664 | 0.925 | 3,980,671 | 0 | False |
| CBAM + quantum-inspired (classical) | `cbam_vqc` | 0.9197 | **0.5792** | 0.5834 | 0.633 | 0.932 | 4,190,663 | 0 | False |
| CBAM + RGVQF | `cbam_rgvqf` | 0.9136 | **0.5735** | 0.5725 | 0.642 | 0.926 | 4,190,759 | 96 | False |
| CBAM + RGVQF, no entanglement | `cbam_rgvqf_no_entangle` | 0.9190 | **0.5848** | 0.5807 | 0.635 | 0.930 | 4,190,759 | 96 | False |
| CBAM + REM + RGVQF | `cbam_rgvqf_rem` | 0.9229 | **0.5853** | 0.5804 | 0.676 | 0.929 | 4,190,789 | 96 | False |
| CBAM + RGVQF encoder, classical circuit (control) | `cbam_rgvqf_classical_ctrl` | 0.9226 | **0.5855** | 0.5779 | 0.663 | 0.927 | 4,190,879 | 0 | False |
| CMAAM + add | `cmaam_add` | 0.9063 | **0.5750** | 0.5769 | 0.621 | 0.931 | 4,202,443 | 0 | False |
| CMAAM + REM + RGVQF | `cmaam_rgvqf_rem` | 0.9183 | **0.5783** | 0.5813 | 0.725 | 0.921 | 4,412,561 | 96 | False |
| CMAAM + REM + RGVQF, P5 only (single-scale) | `cmaam_rgvqf_rem_single_scale` | 0.9313 | **0.5945** | 0.5968 | 0.659 | 0.934 | 4,340,237 | 32 | False |
| Complete (CMAAM + REM + RGVQF + composite loss) | `complete` | 0.9092 | **0.5690** | 0.5725 | 0.597 | 0.927 | 4,412,561 | 96 | False |

P and R are read at confidence 0.25 from the PR curve; mAP is threshold-free. Params used exclude the unused stock backbone copy kept in the checkpoint (1,365,472).

## 2. Paired test: complete vs cbam_add, seeds 0–7 (n = 8)

Same seed ⇒ same data order (B5) and identical COCO-pretrained backbone/neck weights; randomly initialised layers are not shared (different modules consume the RNG differently). Two-sided tests.

### mAP50-95 (primary)

| seed | complete | cbam_add | difference |
|---|---|---|---|
| 0 | 0.5690 | 0.5781 | -0.0091 |
| 1 | 0.5705 | 0.5673 | +0.0031 |
| 2 | 0.5751 | 0.5885 | -0.0134 |
| 3 | 0.5516 | 0.5665 | -0.0149 |
| 4 | 0.5661 | 0.5716 | -0.0055 |
| 5 | 0.5481 | 0.5829 | -0.0349 |
| 6 | 0.5801 | 0.5867 | -0.0067 |
| 7 | 0.5727 | 0.5706 | +0.0021 |

- complete 0.5666 ± 0.0112 · cbam_add 0.5765 ± 0.0087 (mean ± SD)
- mean paired difference **-0.0099** (SD 0.0120), 95% CI [-0.0199, +0.0001]
- paired t(7) = -2.338, **p = 0.0520** · Wilcoxon signed-rank **p = 0.0391** · Cohen's d_z = -0.827 · 2/8 seeds favour complete

### mAP50 (secondary)

| seed | complete | cbam_add | difference |
|---|---|---|---|
| 0 | 0.9092 | 0.9197 | -0.0105 |
| 1 | 0.9176 | 0.9043 | +0.0133 |
| 2 | 0.9219 | 0.9212 | +0.0007 |
| 3 | 0.8863 | 0.8885 | -0.0022 |
| 4 | 0.8933 | 0.9112 | -0.0179 |
| 5 | 0.8978 | 0.9224 | -0.0246 |
| 6 | 0.9145 | 0.9281 | -0.0136 |
| 7 | 0.9189 | 0.8930 | +0.0259 |

- complete 0.9074 ± 0.0133 · cbam_add 0.9111 ± 0.0145 (mean ± SD)
- mean paired difference **-0.0036** (SD 0.0168), 95% CI [-0.0176, +0.0104]
- paired t(7) = -0.609, **p = 0.5615** · Wilcoxon signed-rank **p = 0.5469** · Cohen's d_z = -0.215 · 3/8 seeds favour complete

### mean of in-loop mAP50-95, epochs 10–12 (secondary)

| seed | complete | cbam_add | difference |
|---|---|---|---|
| 0 | 0.5725 | 0.5791 | -0.0066 |
| 1 | 0.5726 | 0.5724 | +0.0003 |
| 2 | 0.5173 | 0.5832 | -0.0658 |
| 3 | 0.5545 | 0.5652 | -0.0108 |
| 4 | 0.5702 | 0.5703 | -0.0001 |
| 5 | 0.5564 | 0.5783 | -0.0219 |
| 6 | 0.5761 | 0.5823 | -0.0062 |
| 7 | 0.5744 | 0.5665 | +0.0079 |

- complete 0.5617 ± 0.0197 · cbam_add 0.5746 ± 0.0070 (mean ± SD)
- mean paired difference **-0.0129** (SD 0.0231), 95% CI [-0.0322, +0.0064]
- paired t(7) = -1.579, **p = 0.1584** · Wilcoxon signed-rank **p = 0.1484** · Cohen's d_z = -0.558 · 2/8 seeds favour complete

**Verdict (primary).** complete − cbam_add = -0.0099 mAP50-95, 95% CI [-0.0199, +0.0001]. Borderline: the two pre-specified tests disagree at α = 0.05 (paired t p = 0.052, Wilcoxon p = 0.039). The direction is consistent: complete is lower than cbam_add in 6/8 seeds. The CI upper bound (+0.0001) rules out a gain for complete larger than that; no analysis in this section shows complete outperforming cbam_add. Secondary metrics (mAP50, mean last 3) are not significant.

Note: complete seed 2 dipped over epochs 10–11 (0.576 → 0.534 → 0.443 → 0.575 at epoch 12). No single-epoch drop exceeded 0.10, so no collapse copy or flag was triggered; per the rule it is reported as-is. It lowers that run's secondary metric (mean last 3 = 0.517), not the primary.

## 3. Collapse summary (27 campaign runs)

- collapse_epoch*.pt files: **0** 
- collapse_log.txt files: **0**
- collapse_flag = True: **0** 
- Near-miss: complete s2, epochs 10–11 (see §2 note). Largest single-epoch drop in any run is listed below.

| run | largest single-epoch drop in in-loop mAP50-95 | at epoch |
|---|---|---|
| abl_complete_s2 | 0.0916 | 11 |
| base_yolo11n_rgb_s0 | 0.0673 | 2 |
| abl_cbam_no_vqc_s6 | 0.0363 | 4 |
| abl_cbam_no_vqc_s3 | 0.0329 | 9 |
| abl_cbam_no_vqc_s2 | 0.0304 | 4 |

## 4. A2 at epoch 12 (RGVQF-type configs; 100 random val images, eval mode, batch 1)

| run | scale | ⟨Z⟩ std | gate std mean / max | saturation (|x|>0.9π) | train-mode saturation | effective gain |γ|/k |
|---|---|---|---|---|---|---|
| abl_cbam_rgvqf_proposed | p3 | 0.325 | 0.0306 / 0.1021 | 0.000 | 0.000 | 0.423 |
| abl_cbam_rgvqf_proposed | p4 | 0.045 | 0.0011 / 0.0032 | 0.010 | 0.000 | 0.513 |
| abl_cbam_rgvqf_proposed | p5 | 0.129 | 0.0026 / 0.0078 | 0.052 | 0.000 | 0.506 |
| abl_cbam_rgvqf_no_entangle | p3 | 0.494 | 0.0210 / 0.0500 | 0.000 | 0.000 | 0.520 |
| abl_cbam_rgvqf_no_entangle | p4 | 0.614 | 0.0276 / 0.0795 | 0.014 | 0.000 | 0.512 |
| abl_cbam_rgvqf_no_entangle | p5 | 0.482 | 0.0115 / 0.0481 | 0.015 | 0.000 | 0.514 |
| abl_cbam_rgvqf_rem | p3 | 0.134 | 0.0065 / 0.0201 | 0.000 | 0.000 | 0.479 |
| abl_cbam_rgvqf_rem | p4 | 0.059 | 0.0027 / 0.0093 | 0.000 | 0.000 | 0.530 |
| abl_cbam_rgvqf_rem | p5 | 0.136 | 0.0035 / 0.0143 | 0.035 | 0.000 | 0.486 |
| abl_cbam_rgvqf_classical_ctrl | p3 | 0.406 | 0.0338 / 0.1014 | 0.000 | 0.000 | 0.519 |
| abl_cbam_rgvqf_classical_ctrl | p4 | 0.567 | 0.0390 / 0.1153 | 0.100 | 0.000 | 0.539 |
| abl_cbam_rgvqf_classical_ctrl | p5 | 0.537 | 0.0224 / 0.0907 | 0.000 | 0.000 | 0.513 |
| abl_cmaam_rgvqf_rem | p3 | 0.062 | 0.0040 / 0.0128 | 0.000 | 0.000 | 0.518 |
| abl_cmaam_rgvqf_rem | p4 | 0.225 | 0.0125 / 0.0400 | 0.000 | 0.000 | 0.523 |
| abl_cmaam_rgvqf_rem | p5 | 0.234 | 0.0072 / 0.0269 | 0.013 | 0.000 | 0.493 |
| abl_mqfp_single_scale | p5 | 0.105 | 0.0068 / 0.0432 | 0.060 | 0.050 | 0.588 |
| abl_complete | p3 | 0.389 | 0.0221 / 0.0678 | 0.135 | 0.000 | 0.388 |
| abl_complete | p4 | 0.216 | 0.0094 / 0.0352 | 0.060 | 0.000 | 0.384 |
| abl_complete | p5 | 0.423 | 0.0253 / 0.0808 | 0.100 | 0.000 | 0.382 |

complete, seeds 0–7 at epoch 12 (mean [min–max] over seeds):

| scale | ⟨Z⟩ std | gate std mean | saturation eval | effective gain |
|---|---|---|---|---|
| p3 | 0.337 [0.250–0.402] | 0.019 [0.015–0.027] | 0.018 [0.000–0.135] | 0.401 [0.388–0.428] |
| p4 | 0.239 [0.043–0.402] | 0.012 [0.002–0.025] | 0.030 [0.000–0.106] | 0.399 [0.360–0.426] |
| p5 | 0.453 [0.207–0.608] | 0.022 [0.010–0.030] | 0.075 [0.000–0.237] | 0.388 [0.317–0.524] |

For the classical control, ⟨Z⟩ is the output of tanh(Linear(8,8)(x)) in place of the circuit.

## 5. Single-factor comparisons (seed 0 only)

**Noise reminder.** Seed-to-seed SD of mAP50-95 within a config is ~0.0100 (complete / cbam_add, n = 8), so a single-seed difference between two configs has SD ≈ 0.0141. **Single-seed gaps under ~0.02 are within noise** and must not be reported as effects.

| factor | A | B | Δ mAP50-95 (A − B) | Δ mean last 3 | Δ mAP50 | within noise (<0.02)? |
|---|---|---|---|---|---|---|
| entanglement on vs off | `cbam_rgvqf` | `cbam_rgvqf_no_entangle` | -0.0114 | -0.0082 | -0.0054 | yes |
| REM on vs off (CBAM + RGVQF) | `cbam_rgvqf_rem` | `cbam_rgvqf` | +0.0118 | +0.0079 | +0.0093 | yes |
| CMAAM vs CBAM (add fusion) | `cmaam_add` | `cbam_add` | -0.0031 | -0.0022 | -0.0134 | yes |
| CMAAM vs CBAM (REM + RGVQF) | `cmaam_rgvqf_rem` | `cbam_rgvqf_rem` | -0.0069 | +0.0009 | -0.0046 | yes |
| multi-scale vs single-scale (P5 only) | `cmaam_rgvqf_rem` | `cmaam_rgvqf_rem_single_scale` | -0.0162 | -0.0155 | -0.0130 | yes |
| composite loss on vs off | `complete` | `cmaam_rgvqf_rem` | -0.0094 | -0.0088 | -0.0090 | yes |
| RGVQF (quantum) vs classical control | `cbam_rgvqf` | `cbam_rgvqf_classical_ctrl` | -0.0120 | -0.0054 | -0.0089 | yes |
| RGVQF vs quantum-inspired (classical) | `cbam_rgvqf` | `cbam_vqc` | -0.0058 | -0.0109 | -0.0060 | yes |
| RGVQF vs add | `cbam_rgvqf` | `cbam_add` | -0.0046 | -0.0066 | -0.0060 | yes |
| quantum-inspired vs add | `cbam_vqc` | `cbam_add` | +0.0011 | +0.0043 | +0.0000 | yes |
| CBAM vs no attention (add) | `cbam_add` | `no_cbam_add` | +0.0123 | +0.0104 | +0.0117 | yes |
| best fusion (cbam_add) vs thermal-only | `cbam_add` | `baseline_thermal` | -0.0007 | -0.0032 | +0.0133 | yes |
| complete vs thermal-only | `complete` | `baseline_thermal` | -0.0098 | -0.0098 | +0.0028 | yes |

Individual seed-0 values are in §1. Spread of the 11 fusion configs at seed 0: mAP50-95 0.5658–0.5945.

## 6. Fusion vs thermal-only, seeds 0–7 (n = 8)

Thermal-only = stock single-stream YOLO11n (COCO-pretrained, 2,590,035 params) on the IR images only, standard Ultralytics trainer with the same fixed validator (768×960) and seeded data order (B5), HSV off and no Albumentations (as the fusion IR stream), otherwise identical settings and 12 epochs. Same seed ⇒ same data order as the fusion runs; the initialisation is not shared (different architecture), so pairing is by seed and data order. All 8 thermal runs: code hash 2dc553937017, 12 epochs, collapse_flag = True: 0, collapse_epoch*.pt files: 0.

### complete vs thermal-only

**mAP50-95 (primary)**

| seed | complete | thermal-only | difference |
|---|---|---|---|
| 0 | 0.5690 | 0.5787 | -0.0098 |
| 1 | 0.5705 | 0.5754 | -0.0049 |
| 2 | 0.5751 | 0.5736 | +0.0014 |
| 3 | 0.5516 | 0.5786 | -0.0270 |
| 4 | 0.5661 | 0.5766 | -0.0105 |
| 5 | 0.5481 | 0.5680 | -0.0199 |
| 6 | 0.5801 | 0.5705 | +0.0095 |
| 7 | 0.5727 | 0.5826 | -0.0099 |

- complete 0.5666 ± 0.0112 · thermal-only 0.5755 ± 0.0047 (mean ± SD)
- mean paired difference **-0.0089** (SD 0.0114), 95% CI [-0.0184, +0.0007]
- paired t(7) = -2.194, **p = 0.0643** · Wilcoxon signed-rank **p = 0.0547** · Cohen's d_z = -0.776 · 2/8 seeds favour complete
- *Primary verdict:* No significant difference at α = 0.05 by either test. The data are compatible with a complete advantage over thermal-only of at most +0.0007 mAP50-95 (upper 95% CI bound).

**mAP50 (secondary)**

| seed | complete | thermal-only | difference |
|---|---|---|---|
| 0 | 0.9092 | 0.9064 | +0.0028 |
| 1 | 0.9176 | 0.9022 | +0.0155 |
| 2 | 0.9219 | 0.9134 | +0.0085 |
| 3 | 0.8863 | 0.9048 | -0.0185 |
| 4 | 0.8933 | 0.8984 | -0.0050 |
| 5 | 0.8978 | 0.8964 | +0.0014 |
| 6 | 0.9145 | 0.9089 | +0.0056 |
| 7 | 0.9189 | 0.9268 | -0.0079 |

- complete 0.9074 ± 0.0133 · thermal-only 0.9071 ± 0.0096 (mean ± SD)
- mean paired difference **+0.0003** (SD 0.0106), 95% CI [-0.0085, +0.0091]
- paired t(7) = 0.080, **p = 0.9388** · Wilcoxon signed-rank **p = 0.8438** · Cohen's d_z = +0.028 · 5/8 seeds favour complete

**mean of in-loop mAP50-95, epochs 10–12 (secondary)**

| seed | complete | thermal-only | difference |
|---|---|---|---|
| 0 | 0.5725 | 0.5823 | -0.0098 |
| 1 | 0.5726 | 0.5725 | +0.0002 |
| 2 | 0.5173 | 0.5742 | -0.0569 |
| 3 | 0.5545 | 0.5747 | -0.0202 |
| 4 | 0.5702 | 0.5769 | -0.0067 |
| 5 | 0.5564 | 0.5716 | -0.0153 |
| 6 | 0.5761 | 0.5694 | +0.0067 |
| 7 | 0.5744 | 0.5835 | -0.0091 |

- complete 0.5617 ± 0.0197 · thermal-only 0.5756 ± 0.0050 (mean ± SD)
- mean paired difference **-0.0139** (SD 0.0193), 95% CI [-0.0300, +0.0022]
- paired t(7) = -2.035, **p = 0.0813** · Wilcoxon signed-rank **p = 0.0547** · Cohen's d_z = -0.719 · 2/8 seeds favour complete

### cbam_add vs thermal-only

**mAP50-95 (primary)**

| seed | cbam_add | thermal-only | difference |
|---|---|---|---|
| 0 | 0.5781 | 0.5787 | -0.0007 |
| 1 | 0.5673 | 0.5754 | -0.0080 |
| 2 | 0.5885 | 0.5736 | +0.0148 |
| 3 | 0.5665 | 0.5786 | -0.0121 |
| 4 | 0.5716 | 0.5766 | -0.0050 |
| 5 | 0.5829 | 0.5680 | +0.0149 |
| 6 | 0.5867 | 0.5705 | +0.0162 |
| 7 | 0.5706 | 0.5826 | -0.0120 |

- cbam_add 0.5765 ± 0.0087 · thermal-only 0.5755 ± 0.0047 (mean ± SD)
- mean paired difference **+0.0010** (SD 0.0124), 95% CI [-0.0093, +0.0114]
- paired t(7) = 0.235, **p = 0.8213** · Wilcoxon signed-rank **p = 0.7422** · Cohen's d_z = +0.083 · 3/8 seeds favour cbam_add
- *Primary verdict:* No significant difference at α = 0.05 by either test. The data are compatible with a cbam_add advantage over thermal-only of at most +0.0114 mAP50-95 (upper 95% CI bound).

**mAP50 (secondary)**

| seed | cbam_add | thermal-only | difference |
|---|---|---|---|
| 0 | 0.9197 | 0.9064 | +0.0133 |
| 1 | 0.9043 | 0.9022 | +0.0021 |
| 2 | 0.9212 | 0.9134 | +0.0078 |
| 3 | 0.8885 | 0.9048 | -0.0163 |
| 4 | 0.9112 | 0.8984 | +0.0129 |
| 5 | 0.9224 | 0.8964 | +0.0260 |
| 6 | 0.9281 | 0.9089 | +0.0192 |
| 7 | 0.8930 | 0.9268 | -0.0337 |

- cbam_add 0.9111 ± 0.0145 · thermal-only 0.9071 ± 0.0096 (mean ± SD)
- mean paired difference **+0.0039** (SD 0.0198), 95% CI [-0.0126, +0.0204]
- paired t(7) = 0.560, **p = 0.5930** · Wilcoxon signed-rank **p = 0.5469** · Cohen's d_z = +0.198 · 6/8 seeds favour cbam_add

**mean of in-loop mAP50-95, epochs 10–12 (secondary)**

| seed | cbam_add | thermal-only | difference |
|---|---|---|---|
| 0 | 0.5791 | 0.5823 | -0.0032 |
| 1 | 0.5724 | 0.5725 | -0.0001 |
| 2 | 0.5832 | 0.5742 | +0.0089 |
| 3 | 0.5652 | 0.5747 | -0.0094 |
| 4 | 0.5703 | 0.5769 | -0.0066 |
| 5 | 0.5783 | 0.5716 | +0.0067 |
| 6 | 0.5823 | 0.5694 | +0.0129 |
| 7 | 0.5665 | 0.5835 | -0.0170 |

- cbam_add 0.5746 ± 0.0070 · thermal-only 0.5756 ± 0.0050 (mean ± SD)
- mean paired difference **-0.0010** (SD 0.0101), 95% CI [-0.0094, +0.0075]
- paired t(7) = -0.274, **p = 0.7921** · Wilcoxon signed-rank **p = 0.8438** · Cohen's d_z = -0.097 · 3/8 seeds favour cbam_add
