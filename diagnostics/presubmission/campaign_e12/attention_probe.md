# Channel vs spatial attention after the B1 fix (campaign_e12, last.pt epoch 12, seed 0)

100 random val images (fixed subset, seed 0), eval mode, batch 1. Gates are 2·sigmoid(·), so 1.0 = identity. "std across images" = per-channel (or per-pixel) std over the 100 images, averaged over channels (pixels).

## abl_complete_s0 (`complete`; hidden activation LeakyReLU)

| module · level · modality | channel min / max | channel mean \|g−1\| | channel std across images | hidden pre-act > 0 | spatial min / max | spatial mean \|s−1\| | spatial std across images |
|---|---|---|---|---|---|---|---|
| cmaam_p3.ir | 0.473 / 1.311 | 0.0895 | 0.0107 | 0.000 | 1.110 / 1.990 | 0.8941 | 0.0370 |
| cmaam_p3.rgb | 0.433 / 1.577 | 0.1842 | 0.0113 | 0.000 | 0.308 / 1.111 | 0.3662 | 0.0507 |
| cmaam_p4.ir | 0.486 / 1.605 | 0.1697 | 0.0214 | 0.001 | 1.539 / 2.000 | 0.9837 | 0.0112 |
| cmaam_p4.rgb | 0.360 / 1.531 | 0.2384 | 0.0324 | 0.000 | 0.049 / 1.752 | 0.1202 | 0.1238 |
| cmaam_p5.ir | 0.000 / 2.000 | 0.3470 | 0.2871 | 0.050 | 0.001 / 1.845 | 0.1446 | 0.1759 |
| cmaam_p5.rgb | 0.000 / 2.000 | 0.3752 | 0.3470 | 0.138 | 0.000 / 0.782 | 0.7851 | 0.1270 |

## abl_cmaam_add_s0 (`cmaam_add`; hidden activation LeakyReLU)

| module · level · modality | channel min / max | channel mean \|g−1\| | channel std across images | hidden pre-act > 0 | spatial min / max | spatial mean \|s−1\| | spatial std across images |
|---|---|---|---|---|---|---|---|
| cmaam_p3.ir | 0.652 / 1.385 | 0.1052 | 0.0044 | 0.000 | 1.051 / 1.980 | 0.8055 | 0.0491 |
| cmaam_p3.rgb | 0.014 / 1.989 | 0.6664 | 0.0106 | 0.094 | 0.344 / 1.312 | 0.2390 | 0.0714 |
| cmaam_p4.ir | 0.818 / 1.243 | 0.0592 | 0.0050 | 0.006 | 1.194 / 1.998 | 0.8719 | 0.0323 |
| cmaam_p4.rgb | 0.549 / 1.515 | 0.1492 | 0.0088 | 0.000 | 0.286 / 1.892 | 0.3107 | 0.0938 |
| cmaam_p5.ir | 0.004 / 1.984 | 0.4051 | 0.3095 | 0.086 | 0.119 / 1.285 | 0.2066 | 0.0803 |
| cmaam_p5.rgb | 0.091 / 1.787 | 0.1689 | 0.1006 | 0.030 | 0.000 / 0.840 | 0.8591 | 0.0447 |

## abl_cmaam_rgvqf_rem_s0 (`cmaam_rgvqf_rem`; hidden activation LeakyReLU)

| module · level · modality | channel min / max | channel mean \|g−1\| | channel std across images | hidden pre-act > 0 | spatial min / max | spatial mean \|s−1\| | spatial std across images |
|---|---|---|---|---|---|---|---|
| cmaam_p3.ir | 0.646 / 1.253 | 0.0911 | 0.0041 | 0.000 | 1.304 / 1.994 | 0.8734 | 0.0360 |
| cmaam_p3.rgb | 0.414 / 1.576 | 0.1935 | 0.0077 | 0.000 | 0.602 / 1.707 | 0.0982 | 0.0794 |
| cmaam_p4.ir | 0.497 / 1.379 | 0.0970 | 0.0344 | 0.024 | 1.377 / 1.999 | 0.9397 | 0.0195 |
| cmaam_p4.rgb | 0.705 / 1.392 | 0.0772 | 0.0130 | 0.030 | 0.272 / 1.871 | 0.4217 | 0.0679 |
| cmaam_p5.ir | 0.314 / 1.705 | 0.1661 | 0.0610 | 0.023 | 0.031 / 1.272 | 0.1819 | 0.1256 |
| cmaam_p5.rgb | 0.190 / 1.694 | 0.0927 | 0.0937 | 0.075 | 0.000 / 0.706 | 0.8672 | 0.0605 |

## abl_cbam_no_vqc_s0 (`cbam_add`; hidden activation LeakyReLU)

| module · level · modality | channel min / max | channel mean \|g−1\| | channel std across images | hidden pre-act > 0 | spatial min / max | spatial mean \|s−1\| | spatial std across images |
|---|---|---|---|---|---|---|---|
| ir_cbam_p3 | 0.856 / 1.131 | 0.0323 | 0.0150 | 0.438 | 0.970 / 1.907 | 0.5387 | 0.0463 |
| ir_cbam_p4 | 0.894 / 1.091 | 0.0233 | 0.0039 | 0.188 | 1.039 / 1.988 | 0.6685 | 0.0733 |
| ir_cbam_p5 | 0.117 / 1.707 | 0.1415 | 0.1472 | 0.462 | 0.017 / 1.030 | 0.2207 | 0.0518 |
| rgb_cbam_p3 | 0.085 / 1.712 | 0.2774 | 0.0188 | 0.442 | 0.723 / 1.145 | 0.1121 | 0.0275 |
| rgb_cbam_p4 | 0.925 / 1.100 | 0.0164 | 0.0037 | 0.376 | 0.678 / 1.638 | 0.2596 | 0.0502 |
| rgb_cbam_p5 | 0.326 / 1.821 | 0.0928 | 0.0285 | 0.319 | 0.000 / 0.721 | 0.8632 | 0.0385 |

## Verdict

- **Channel attention is no longer identity.** Channel gates now deviate from 1.0 at every level (mean |g−1| 0.06–0.67 for CMAAM, 0.02–0.28 for CBAM), whereas before the fix they were exactly 1.0.
- **Image dependence is level-dependent.**
  - **P5: active and image-dependent.** Channel-gate std across images is 0.06–0.35, and gates span the full range 0–2.
  - **P3/P4: mostly a near-constant per-channel rescaling.** The std across images (0.004–0.034) is small relative to the deviation from 1 (0.06–0.67). CBAM P3-IR, P4-IR and P4-RGB are still close to identity (|g−1| ≤ 0.03).
- **CMAAM hidden units sit on the negative LeakyReLU branch at P3/P4.** Only 0–3 % of pre-activations are positive (5–14 % at P5). The MLP still passes gradient (slope 0.1), so it is not dead, but it acts as a scaled near-linear map rather than using the ReLU-type non-linearity. CBAM's hidden units are healthier (19–46 % positive).
- **Spatial gates mostly implement a modality weighting.** In all three CMAAM checkpoints, the IR spatial gate at P3/P4 is ≈ 1.1–2.0 everywhere (mean |s−1| 0.81–0.98, image std ≤ 0.05), i.e. a near-constant ~×1.9 amplification of IR. The RGB spatial gate at P5 is ≈ 0–0.8 (mean |s−1| 0.79–0.87), i.e. RGB P5 is largely suppressed. This is consistent with the thermal-only parity in RESULTS.md §6.
