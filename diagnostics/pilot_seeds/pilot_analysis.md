# Variance pilot -- paired multi-seed, proposed vs baseline

Proposed: `abl_cbam_rgvqf_proposed`  Baseline: `abl_cbam_no_vqc`  Seeds: [0, 1, 2, 3, 4]

Seed 0 reused from `diagnostics/step5_retrain/ranked_grid.csv` (not retrained); seeds 1-4 trained here through `run_step5_retrain.train_config()`/`revalidate_checkpoint()` unmodified, EPOCHS=50 PATIENCE=20 (byte-identical to the grid's hyperparameters).

## Per-seed values and paired diffs

| seed | proposed mAP50 | baseline mAP50 | diff mAP50 | proposed mAP50-95 | baseline mAP50-95 | diff mAP50-95 |
|---|---|---|---|---|---|---|
| 0 | 0.8456 | 0.8487 | -0.0032 | 0.3402 | 0.3329 | +0.0073 |
| 1 | 0.8321 | 0.8141 | +0.0180 | 0.3427 | 0.3325 | +0.0101 |
| 2 | 0.8073 | 0.8290 | -0.0216 | 0.3338 | 0.3361 | -0.0023 |
| 3 | 0.8130 | 0.8423 | -0.0293 | 0.3391 | 0.3311 | +0.0080 |
| 4 | 0.8266 | 0.8243 | +0.0022 | 0.3343 | 0.3295 | +0.0049 |

## mAP50 statistics

- n = 5
- mean diff = -0.0068
- SD of diffs (measured seed-to-seed variance) = 0.0189
- paired t = -0.800, df = 4
- two-sided p-value = 0.4684
- Cohen's dz = -0.358

## mAP50-95 statistics (PRIMARY decision metric)

- n = 5
- mean diff = +0.0056
- SD of diffs (measured seed-to-seed variance) = 0.0048
- paired t = 2.592, df = 4
- two-sided p-value = 0.0606
- Cohen's dz = 1.159

## Power: seeds/config required for 80% power at the MEASURED effect size

- measured dz (mAP50-95) = 1.159 -> n = 8

## Verdict logic

- SIGNIFICANT: primary (mAP50-95) p < 0.05 two-sided AND mean diff > 0 (proposed wins).
- UNDERPOWERED-BUT-PROMISING: not significant, but all 5 per-seed diffs share the same sign AND |Cohen's dz| >= 0.5 (a moderate-or-larger measured effect that this n=5 study isn't powered to confirm) -- N below is the seeds/config needed for 80% power at the measured effect size.
- PARITY: neither of the above -- either the diffs don't share a consistent sign, or the measured effect is too small to be worth chasing with more seeds.

## VERDICT: PARITY

No extension, no third config, launched from this pilot. Hand-back only.
