# B6 pilot — cbam_rgvqf, seed 0, 5 epochs, B1+B2+B3+B5 ON

| variant | epochs | mAP50 (last.pt reval) | mAP50-95 (last.pt reval) | P@.25 | R@.25 |
|---|---|---|---|---|---|
| b6_off | 5 | 0.9167 | 0.5929 | 0.553 | 0.943 |
| b6_on | 5 | 0.9250 | 0.5778 | 0.626 | 0.929 |

| scale | <Z> std OFF | <Z> std ON | gate std OFF (mean / max ch) | gate std ON (mean / max ch) | gate-std ratio ON/OFF |
|---|---|---|---|---|---|
| p3 | 0.0080 | 0.2834 | 0.00023 / 0.00090 | 0.01051 / 0.03311 | 45.4x |
| p4 | 0.0077 | 0.2958 | 0.00019 / 0.00059 | 0.00745 / 0.02207 | 39.2x |
| p5 | 0.0040 | 0.2755 | 0.00012 / 0.00036 | 0.00347 / 0.01330 | 30.0x |

- Criterion 1 (gate std >= 10x at every scale): **PASS** (ratios [45.4, 39.2, 30.0])
- Criterion 2 (B6 ON mAP50-95 not lower): **FAIL** (Δ mAP50-95 = -0.0151, Δ mAP50 = +0.0083; single seed — seed-to-seed SD of paired diffs at 50 ep was ~0.018)

## Addendum

- Both training runs used the grid settings (`runs/*/args.yaml` and `manifest.json`): box 10, cls 0.3, dfl 1.5, mosaic/mixup/copy_paste/degrees/translate/scale/shear/perspective/flipud/fliplr 0, close_mosaic 0, epochs 5, rect, imgsz 960. The default-valued trainer banners in run.log (`reval_*`, `a2`) are post-training evaluation helpers that build only a val loader. Val transforms are identical under default vs grid aug hyperparameters: **True** — compared by transform type and parameters: LetterBox(new_shape 960, auto False, scaleup False, center, pad 114) + Format. (An earlier automated line said False because it compared repr() strings containing memory addresses; corrected.)
- The A2 numbers in the table above were measured in **eval mode with batch 4**. Re-measured below in eval mode with batch 1 (must equal batch 4, since eval mode has no batch coupling), and with the whole model in **train mode** (batch 4, BN batch stats).

| run | scale | <Z> std eval b1 | eval b4 | train b4 | gate std (mean / max ch) eval b1 | train b4 | |x|>0.9π eval b1 | train b4 |
|---|---|---|---|---|---|---|---|---|
| b6_off | p3 | 0.0080 | 0.0080 | 0.0045 | 0.00023 / 0.00090 | 0.00013 / 0.00051 | 0.0000 | 0.0000 |
| b6_off | p4 | 0.0077 | 0.0077 | 0.0039 | 0.00019 / 0.00059 | 0.00010 / 0.00030 | 0.0000 | 0.0000 |
| b6_off | p5 | 0.0040 | 0.0040 | 0.0054 | 0.00012 / 0.00036 | 0.00016 / 0.00049 | 0.0000 | 0.0000 |
| b6_on | p3 | 0.2834 | 0.2834 | 0.2317 | 0.01051 / 0.03311 | 0.00650 / 0.01939 | 0.6450 | 0.0463 |
| b6_on | p4 | 0.2958 | 0.2958 | 0.2071 | 0.00745 / 0.02207 | 0.00419 / 0.01076 | 0.4062 | 0.0487 |
| b6_on | p5 | 0.2755 | 0.2755 | 0.2742 | 0.00347 / 0.01330 | 0.00453 / 0.01718 | 0.1887 | 0.1350 |

Encoder BatchNorm running_var / running_mean (b6_on, per scale, 8 dims):

- p3: var [0.0032, 0.0032, 0.005, 0.0058, 0.0069, 0.0024, 0.0072, 0.0063] · mean [0.3347, -0.438, -0.0454, 0.5947, -0.5244, 0.3774, 0.4045, -0.3882]
- p4: var [0.0067, 0.0053, 0.0092, 0.0079, 0.0099, 0.0112, 0.0111, 0.0093] · mean [0.072, -0.2267, 0.1954, -0.0417, 0.1791, -0.0111, -0.3757, 0.1691]
- p5: var [0.0252, 0.0181, 0.032, 0.0181, 0.0167, 0.0167, 0.0129, 0.0183] · mean [-0.2742, -0.1055, -0.2981, -0.1931, -0.0114, -0.2871, 0.0615, -0.0105]

Per-epoch in-loop mAP50 / mAP50-95 (fixed validator, EMA weights):

| epoch | b6_off mAP50 | b6_off mAP50-95 | b6_on mAP50 | b6_on mAP50-95 |
|---|---|---|---|---|
| 1 | 0.8736 | 0.4993 | 0.8652 | 0.5066 |
| 2 | 0.9130 | 0.5513 | 0.8740 | 0.5231 |
| 3 | 0.9260 | 0.5579 | 0.9207 | 0.5564 |
| 4 | 0.9134 | 0.5905 | 0.9183 | 0.5830 |
| 5 | 0.9166 | 0.5930 | 0.9249 | 0.5778 |

### Eval-mode saturation in B6 ON: cause

On 100 val images, the encoder's pre-BN output has population variance 0.0010/0.0015/0.0070 (P3/P4/P5); BN running_var is 0.0050/0.0088/0.0197, and the val mean differs from running_mean by 0.11/0.12/0.08 per dim (~1.5 running SDs). In eval the BN therefore maps val samples to an offset distribution (effective SD 1.65/1.38/0.84 before the affine), and pi*tanh saturates 64%/41%/19% of angles (|x|>0.9pi), vs 5%/5%/14% in train mode (batch stats). The batch-of-4 variance hypothesis is rejected; the cause is the train-vs-val running_mean offset amplified by the ~15-30x normalisation the tiny pooled signal needs. Train/eval mismatch of B6 as specified.
