# Phase 2 — re-validation with fixed val geometry (768x960, pad 0)
Rows: 48 / 48 expected.
## Change in mAP50-95 (rows with a pre-fix value)
- n = 34; mean change = +0.2305; min = +0.2055; max = +0.2627
- max |change| = 0.2627 (abl_cbam_no_vqc seed 7 best.pt: 0.3269 -> 0.5896)

## 10-config ranking, best.pt (by mAP50-95)

Ranking changed: **True**; Spearman rho(old, new) = 0.358

| new rank | config | mAP50 new | mAP50-95 new | P@.25 | R@.25 | old rank | mAP50-95 old |
|---|---|---|---|---|---|---|---|
| 1 | abl_cbam_rgvqf_proposed | 0.9256 | 0.5799 | 0.659 | 0.926 | 1 | 0.3402 |
| 2 | abl_cmaam_rgvqf_rem | 0.9260 | 0.5782 | 0.741 | 0.907 | 3 | 0.3399 |
| 3 | abl_cbam_no_vqc | 0.9130 | 0.5768 | 0.653 | 0.923 | 8 | 0.3329 |
| 4 | abl_no_cbam_add | 0.9179 | 0.5751 | 0.707 | 0.913 | 2 | 0.3401 |
| 5 | abl_mqfp_single_scale | 0.9173 | 0.5719 | 0.670 | 0.920 | 9 | 0.3321 |
| 6 | abl_cbam_rgvqf_rem | 0.9084 | 0.5690 | 0.636 | 0.922 | 10 | 0.3276 |
| 7 | abl_cbam_vqc_proposed | 0.9094 | 0.5653 | 0.674 | 0.917 | 4 | 0.3381 |
| 8 | abl_cmaam_add | 0.8915 | 0.5567 | 0.623 | 0.908 | 7 | 0.3339 |
| 9 | abl_cbam_rgvqf_no_entangle | 0.9188 | 0.5554 | 0.635 | 0.924 | 6 | 0.3347 |
| 10 | abl_complete | 0.8960 | 0.5546 | 0.633 | 0.917 | 5 | 0.3367 |

Spread of new mAP50-95 across configs: 0.5546–0.5799 (range 0.0253)

## 10-config ranking, last.pt (by mAP50-95)

Ranking changed: **True**; Spearman rho(old, new) = -0.297

| new rank | config | mAP50 new | mAP50-95 new | P@.25 | R@.25 | old rank | mAP50-95 old |
|---|---|---|---|---|---|---|---|
| 1 | abl_cbam_rgvqf_no_entangle | 0.9030 | 0.5604 | 0.607 | 0.925 | 7 | 0.3210 |
| 2 | abl_cbam_no_vqc | 0.9025 | 0.5583 | 0.628 | 0.917 | 8 | 0.3180 |
| 3 | abl_mqfp_single_scale | 0.8928 | 0.5576 | 0.684 | 0.905 | 6 | 0.3244 |
| 4 | abl_cmaam_add | 0.8915 | 0.5567 | 0.623 | 0.908 | 3 | 0.3339 |
| 5 | abl_cbam_rgvqf_rem | 0.8875 | 0.5532 | 0.638 | 0.907 | 10 | 0.3141 |
| 6 | abl_cmaam_rgvqf_rem | 0.8944 | 0.5505 | 0.648 | 0.903 | 4 | 0.3320 |
| 7 | abl_complete | 0.8782 | 0.5441 | 0.616 | 0.908 | 5 | 0.3315 |
| 8 | abl_no_cbam_add | 0.8662 | 0.5420 | 0.598 | 0.909 | 1 | 0.3360 |
| 9 | abl_cbam_vqc_proposed | 0.8793 | 0.5400 | 0.640 | 0.907 | 2 | 0.3345 |
| 10 | abl_cbam_rgvqf_proposed | 0.8736 | 0.5334 | 0.608 | 0.917 | 9 | 0.3146 |

Spread of new mAP50-95 across configs: 0.5334–0.5604 (range 0.0271)

## 8-seed paired test: proposed (cbam+rgvqf) vs baseline (cbam+add)

### best.pt — mAP50-95 (n=8, seeds [0, 1, 2, 3, 4, 5, 6, 7])

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.5799 | 0.5768 | +0.0030 |
| 1 | 0.5839 | 0.5621 | +0.0219 |
| 2 | 0.5452 | 0.5685 | -0.0234 |
| 3 | 0.5468 | 0.5693 | -0.0226 |
| 4 | 0.5748 | 0.5426 | +0.0322 |
| 5 | 0.5828 | 0.5722 | +0.0106 |
| 6 | 0.5826 | 0.5648 | +0.0178 |
| 7 | 0.5587 | 0.5896 | -0.0309 |

- proposed 0.5693 ± 0.0165; baseline 0.5682 ± 0.0134
- mean diff +0.0011 (SD 0.0238); paired t = 0.128, p = 0.9021; Wilcoxon p = 1.0000; Cohen's dz = 0.045; 5/8 seeds favour proposed

### best.pt — mAP50 (n=8, seeds [0, 1, 2, 3, 4, 5, 6, 7])

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.9256 | 0.9130 | +0.0125 |
| 1 | 0.9214 | 0.9052 | +0.0162 |
| 2 | 0.8822 | 0.9114 | -0.0293 |
| 3 | 0.8915 | 0.9283 | -0.0368 |
| 4 | 0.9248 | 0.8540 | +0.0708 |
| 5 | 0.9357 | 0.9243 | +0.0113 |
| 6 | 0.9186 | 0.9059 | +0.0127 |
| 7 | 0.8978 | 0.9318 | -0.0340 |

- proposed 0.9122 ± 0.0191; baseline 0.9092 ± 0.0245
- mean diff +0.0029 (SD 0.0359); paired t = 0.231, p = 0.8236; Wilcoxon p = 1.0000; Cohen's dz = 0.082; 5/8 seeds favour proposed

### last.pt — mAP50-95 (n=8, seeds [0, 1, 2, 3, 4, 5, 6, 7])

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.5334 | 0.5583 | -0.0249 |
| 1 | 0.5537 | 0.5505 | +0.0031 |
| 2 | 0.5452 | 0.5447 | +0.0004 |
| 3 | 0.5448 | 0.5577 | -0.0129 |
| 4 | 0.5563 | 0.5246 | +0.0317 |
| 5 | 0.5480 | 0.5281 | +0.0199 |
| 6 | 0.5601 | 0.5526 | +0.0076 |
| 7 | 0.5604 | 0.5572 | +0.0032 |

- proposed 0.5502 ± 0.0092; baseline 0.5467 ± 0.0134
- mean diff +0.0035 (SD 0.0176); paired t = 0.566, p = 0.5889; Wilcoxon p = 0.4609; Cohen's dz = 0.200; 6/8 seeds favour proposed

### last.pt — mAP50 (n=8, seeds [0, 1, 2, 3, 4, 5, 6, 7])

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.8736 | 0.9025 | -0.0289 |
| 1 | 0.8818 | 0.8950 | -0.0132 |
| 2 | 0.8822 | 0.8869 | -0.0048 |
| 3 | 0.8885 | 0.9112 | -0.0227 |
| 4 | 0.9072 | 0.8331 | +0.0741 |
| 5 | 0.9100 | 0.8835 | +0.0266 |
| 6 | 0.9011 | 0.8828 | +0.0183 |
| 7 | 0.9057 | 0.8876 | +0.0182 |

- proposed 0.8938 ± 0.0139; baseline 0.8853 ± 0.0233
- mean diff +0.0084 (SD 0.0335); paired t = 0.713, p = 0.4986; Wilcoxon p = 0.7422; Cohen's dz = 0.252; 4/8 seeds favour proposed

## Per-run training history (in-loop selection used the OLD misaligned validator)

| run | seed | epochs trained | best epoch (in-loop) | early-stopped (<50) | best.pt mAP50-95 (fixed) | last.pt mAP50-95 (fixed) | last − best | s/epoch |
|---|---|---|---|---|---|---|---|---|
| abl_cbam_no_vqc | 0 | 26 | 6 | yes | 0.5768 | 0.5583 | -0.0186 | 1188.7 |
| abl_cbam_rgvqf_no_entangle | 0 | 22 | 2 | yes | 0.5554 | 0.5604 | +0.0050 | 1255.4 |
| abl_cbam_rgvqf_proposed | 0 | 34 | 14 | yes | 0.5799 | 0.5334 | -0.0465 | 1276.6 |
| abl_cbam_rgvqf_rem | 0 | 37 | 17 | yes | 0.5690 | 0.5532 | -0.0158 | 1708.9 |
| abl_cbam_vqc_proposed | 0 | 47 | 27 | yes | 0.5653 | 0.5400 | -0.0253 | 1162.4 |
| abl_cmaam_add | 0 | 50 | 50 | no | 0.5567 | 0.5567 | +0.0000 | 1101.4 |
| abl_cmaam_rgvqf_rem | 0 | 36 | 16 | yes | 0.5782 | 0.5505 | -0.0277 | 1323.7 |
| abl_complete | 0 | 50 | 36 | no | 0.5546 | 0.5441 | -0.0105 | 1328.7 |
| abl_mqfp_single_scale | 0 | 40 | 20 | yes | 0.5719 | 0.5576 | -0.0143 | 1179.2 |
| abl_no_cbam_add | 0 | 48 | 28 | yes | 0.5751 | 0.5420 | -0.0331 | 995.3 |
| abl_cbam_rgvqf_proposed | 1 | 35 | 15 | yes | 0.5839 | 0.5537 | -0.0303 | 1318.8 |
| abl_cbam_no_vqc | 1 | 49 | 29 | yes | 0.5621 | 0.5505 | -0.0115 | 1145.4 |
| abl_cbam_rgvqf_proposed | 2 | 50 | 50 | no | 0.5452 | 0.5452 | +0.0000 | 1276.0 |
| abl_cbam_no_vqc | 2 | 36 | 16 | yes | 0.5685 | 0.5447 | -0.0238 | 1124.2 |
| abl_cbam_rgvqf_proposed | 3 | 50 | 42 | no | 0.5468 | 0.5448 | -0.0020 | 1259.5 |
| abl_cbam_no_vqc | 3 | 26 | 6 | yes | 0.5693 | 0.5577 | -0.0117 | 1150.2 |
| abl_cbam_rgvqf_proposed | 4 | 36 | 16 | yes | 0.5748 | 0.5563 | -0.0185 | 1260.4 |
| abl_cbam_no_vqc | 4 | 43 | 23 | yes | 0.5426 | 0.5246 | -0.0180 | 1115.1 |
| abl_cbam_rgvqf_proposed | 5 | 31 | 11 | yes | 0.5828 | 0.5480 | -0.0348 | 1359.4 |
| abl_cbam_no_vqc | 5 | 36 | 16 | yes | 0.5722 | 0.5281 | -0.0442 | 1102.7 |
| abl_cbam_rgvqf_proposed | 6 | 33 | 13 | yes | 0.5826 | 0.5601 | -0.0224 | 1275.9 |
| abl_cbam_no_vqc | 6 | 42 | 22 | yes | 0.5648 | 0.5526 | -0.0122 | 1126.9 |
| abl_cbam_rgvqf_proposed | 7 | 36 | 16 | yes | 0.5587 | 0.5604 | +0.0017 | 1286.1 |
| abl_cbam_no_vqc | 7 | 33 | 13 | yes | 0.5896 | 0.5572 | -0.0324 | 1101.8 |

- Early-stopped before epoch 50: 20/24 runs. Median epochs trained: 36.0; median in-loop best epoch: 16.0.
- last.pt − best.pt (fixed geometry): mean -0.0186, range -0.0465 … +0.0050.
- Measured wall-clock per epoch (train + in-loop val, MPS), median per config: abl_cbam_no_vqc 1126s; abl_cbam_rgvqf_no_entangle 1255s; abl_cbam_rgvqf_proposed 1276s; abl_cbam_rgvqf_rem 1709s; abl_cbam_vqc_proposed 1162s; abl_cmaam_add 1101s; abl_cmaam_rgvqf_rem 1324s; abl_complete 1329s; abl_mqfp_single_scale 1179s; abl_no_cbam_add 995s

## Retrain options (compute estimates on this M1 Pro, MPS, sequential)

Basis: measured median wall-clock per epoch from each run's results.csv (train + in-loop val),
from phase2_runs.csv: cbam_add 1126 s, no_cbam_add 995, cbam_vqc 1162, cbam_rgvqf 1276,
no_entangle 1255, cbam_rgvqf_rem 1709 (single run, likely inflated), cmaam_add 1101,
cmaam_rgvqf_rem 1324, single_scale 1179, complete 1329. Sum over 10 configs = 12,456 s/epoch.
In-loop val ≈ 2–2.5 min of each epoch (Phase 2 smoke: 142 s per full revalidation incl. setup).

| option | what | runs | est. wall-clock |
|---|---|---|---|
| O1 | No retrain. Report fixed-geometry re-validation of existing checkpoints; disclose that early stopping / best.pt used the misaligned validator | 0 | 0 |
| O2 (requested) | Retrain all 10 configs, fixed validator, **fixed 50 epochs (patience=50), report last.pt**, seed 0 | 10 | 173 h ≈ 7.2 days |
| O2-noval | O2 with `val=False` (Ultralytics still validates once after the final epoch); no per-epoch curves | 10 | ≈ 155 h ≈ 6.5 days |
| O2 + key-pair seeds | O2 + seeds 1–7 for abl_complete and abl_cbam_no_vqc (Phase 4 claim), same protocol | 10 + 14 | 173 + 239 h ≈ 17 days |
| O2 + 3-way seeds | O2 + seeds 1–7 for complete, cbam_rgvqf, cbam_add | 10 + 21 | 173 + 328 h ≈ 21 days |
| O3 | O2 protocol but 30 epochs (median in-loop best epoch was 16) | 10 | 104 h ≈ 4.3 days |
| Phase 3 | thermal-only and RGB-only single-stream baselines, 50 ep | 2 | UNVERIFIED until smoke test; single stream ≈ half the backbone cost |

Power at fixed geometry (paired SD of diffs, two-sided α=.05, 80% power):
best.pt SD 0.0238 → minimum detectable mean diff 0.0275 (n=8), 0.0178 (n=16);
last.pt SD 0.0176 → 0.0203 (n=8), 0.0156 (n=12), 0.0132 (n=16).
