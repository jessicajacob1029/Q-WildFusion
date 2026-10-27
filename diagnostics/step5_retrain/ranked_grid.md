# Step 5 retrain -- ranked grid (mAP50-95 descending)

MODERATE_EPOCHS=50  PATIENCE=20  SEED=0

| config | checkpoint | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| abl_cbam_rgvqf_proposed | best.pt | 0.689 | 0.858 | 0.846 | 0.340 |
| abl_no_cbam_add | best.pt | 0.644 | 0.841 | 0.826 | 0.340 |
| abl_cmaam_rgvqf_rem | best.pt | 0.761 | 0.835 | 0.844 | 0.340 |
| abl_cbam_vqc_proposed | best.pt | 0.640 | 0.836 | 0.827 | 0.338 |
| abl_complete | best.pt | 0.657 | 0.839 | 0.824 | 0.337 |
| abl_no_cbam_add | last.pt | 0.601 | 0.832 | 0.813 | 0.336 |
| abl_cbam_rgvqf_no_entangle | best.pt | 0.582 | 0.869 | 0.845 | 0.335 |
| abl_cbam_vqc_proposed | last.pt | 0.615 | 0.829 | 0.809 | 0.334 |
| abl_cmaam_add | best.pt | 0.661 | 0.831 | 0.815 | 0.334 |
| abl_cmaam_add | last.pt | 0.661 | 0.831 | 0.815 | 0.334 |
| abl_cbam_no_vqc | best.pt | 0.581 | 0.860 | 0.849 | 0.333 |
| abl_mqfp_single_scale | best.pt | 0.644 | 0.840 | 0.822 | 0.332 |
| abl_cmaam_rgvqf_rem | last.pt | 0.722 | 0.837 | 0.823 | 0.332 |
| abl_complete | last.pt | 0.607 | 0.832 | 0.804 | 0.332 |
| abl_cbam_rgvqf_rem | best.pt | 0.675 | 0.839 | 0.829 | 0.328 |
| abl_mqfp_single_scale | last.pt | 0.620 | 0.824 | 0.793 | 0.324 |
| abl_cbam_rgvqf_no_entangle | last.pt | 0.635 | 0.844 | 0.823 | 0.321 |
| abl_cbam_no_vqc | last.pt | 0.615 | 0.836 | 0.809 | 0.318 |
| abl_cbam_rgvqf_proposed | last.pt | 0.678 | 0.838 | 0.811 | 0.315 |
| abl_cbam_rgvqf_rem | last.pt | 0.628 | 0.827 | 0.803 | 0.314 |

## Flagged -- not acted on, hand back only

### (a) still climbing at the epoch cap (undertrained, extension candidate)

- abl_cmaam_add

### (b) decision-critical gaps smaller than the configs' own best<->last spread (within noise, needs seeds/extension before it can be claimed)

- **proposed (cbam+rgvqf) vs baseline (cbam+add)** (abl_cbam_rgvqf_proposed vs abl_cbam_no_vqc): gap=0.0073 < noise floor=0.0256
- **entangle vs no-entangle (rgvqf)** (abl_cbam_rgvqf_proposed vs abl_cbam_rgvqf_no_entangle): gap=0.0055 < noise floor=0.0256
- **complete vs proposed** (abl_complete vs abl_cbam_rgvqf_proposed): gap=0.0035 < noise floor=0.0256

No extensions or multi-seed runs were launched. This is a hand-back for a decision, not an auto-continuation.
