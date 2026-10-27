# n=8 extension -- paired multi-seed, proposed vs baseline

Proposed: `abl_cbam_rgvqf_proposed`  Baseline: `abl_cbam_no_vqc`  Seeds: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)

Seeds 0-4 pooled from `diagnostics/pilot_seeds/pilot_results.csv` (n=5 pilot, not retrained); seeds 5-7 trained here through `run_pilot_seeds.train_and_validate()` -> `run_step5_retrain.train_config()`/`revalidate_checkpoint()` unmodified. EPOCHS=50 PATIENCE=20 deterministic=True -- byte-identical to seeds 0-4.

## mAP50-95 statistics (PRIMARY -- verdict decided on this)

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.3402 | 0.3329 | +0.0073 |
| 1 | 0.3427 | 0.3325 | +0.0101 |
| 2 | 0.3338 | 0.3361 | -0.0023 |
| 3 | 0.3391 | 0.3311 | +0.0080 |
| 4 | 0.3343 | 0.3295 | +0.0049 |
| 5 | 0.3251 | 0.3357 | -0.0106 |
| 6 | 0.3355 | 0.3445 | -0.0090 |
| 7 | 0.3240 | 0.3269 | -0.0030 |

- n = 8
- mean diff = +0.0007
- SD of diffs = 0.0080
- 95% CI of mean diff (t, df=7) = [-0.0060, +0.0074]
- paired t = 0.236, df = 7, two-sided p = 0.8202
- Cohen's dz = 0.083
- sign consistency = 4/8 seeds favor the mean's direction (proposed)

## mAP50 statistics

| seed | proposed | baseline | diff |
|---|---|---|---|
| 0 | 0.8456 | 0.8487 | -0.0032 |
| 1 | 0.8321 | 0.8141 | +0.0180 |
| 2 | 0.8073 | 0.8290 | -0.0216 |
| 3 | 0.8130 | 0.8423 | -0.0293 |
| 4 | 0.8266 | 0.8243 | +0.0022 |
| 5 | 0.8393 | 0.8383 | +0.0010 |
| 6 | 0.8293 | 0.8339 | -0.0046 |
| 7 | 0.8201 | 0.8319 | -0.0117 |

- n = 8
- mean diff = -0.0062
- SD of diffs = 0.0147
- 95% CI of mean diff (t, df=7) = [-0.0185, +0.0062]
- paired t = -1.182, df = 7, two-sided p = 0.2756
- Cohen's dz = -0.418
- sign consistency = 5/8 seeds favor the mean's direction (baseline)

## Power at the n=8 measured effect

- effect size zero/undefined -- no finite N reaches 80% power

## Verdict logic (fixed -- primary metric mAP50-95)

- SIGNIFICANT_PROPOSED : p < 0.05 AND mean diff > 0
- SIGNIFICANT_BASELINE : p < 0.05 AND mean diff < 0
- PROMISING_UNDERPOWERED : p >= 0.05 AND |dz| >= 0.5 AND >= 6/8 seeds share the mean's sign
- PARITY : p >= 0.05 AND |dz| < 0.5  (also: p>=0.05 with large dz but <6/8 sign agreement)

## VERDICT (mAP50-95): PARITY

- p = 0.8202, mean diff = +0.0007, |dz| = 0.083, sign agreement = 4/8

## mAP50 (secondary -- stated separately so the metrics aren't conflated)

- p = 0.2756, mean diff = -0.0062, |dz| = 0.418, sign agreement = 5/8. Parity on mAP50.

Hand-back only. No further seeds, no third config launched from this run.
