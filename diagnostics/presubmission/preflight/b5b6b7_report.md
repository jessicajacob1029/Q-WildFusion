# B5 / B6 / B7 — report

Code diff (all pre-flight changes against HEAD): `all_code_changes.diff`.
Raw outputs: `b5.json`, `b6_diagnose.json`, `b6_fix.json`, and `b5.log` / `b6diag.log` / `b6fix.log` / `b7.log`.

## Incident: dataset cache overwritten, then restored

- The Part B short runs used `fraction=0.05` against `phase3/dataset_aligned`. Ultralytics' label-cache hash then no longer matched, so it rewrote `labels/rgb/train.cache` (1.6 MB → 90 KB).
- No label `.txt` files were touched, and Ultralytics would have rebuilt the cache on the next full run.
- **Restored** byte-identical from HEAD via `git show` (SHA-256 `8769e19c…` = HEAD). The overwritten copy is kept as `train.cache.overwritten_by_fraction_run`.
- **Prevention.**
  - `check_det_dataset` calls `.resolve()` on the train/val paths (`ultralytics/data/utils.py:458`), so directory symlinks lead back into `dataset_aligned`.
  - Every run with `fraction<1` (Part D smoke tests) and every Phase 3 baseline (different file lists → different hash) therefore needs a dataset folder of **per-file symlinks in real directories** under `diagnostics/presubmission/`. The caches are then written there.
  - All checks in this report ran at `fraction=1.0`. `git status phase3/dataset_aligned` is clean.

## B5 — seeded data order (flag `seeded_data_order`; fusion trainer + `diagnostics/presubmission/baseline_trainer.py`)

| check | result |
|---|---|
| fusion, flag OFF, seed 0 == old order (A4) | PASS |
| fusion, flag OFF, seed 0 == seed 1 (old behaviour) | PASS |
| fusion, flag ON, seed 0 ≠ seed 1 | PASS (batch 1: `100179,010361,010325,100221` vs `140499,040444,080540,090337`) |
| baseline, flag OFF, seed 0 == old order | PASS |
| baseline, flag ON, seed 0 ≠ seed 1 | PASS |
| fusion ON seed s == baseline ON seed s | PASS (paired comparisons see identical data order) |

## B6-diagnose — why x is image-independent (100 val images, last.pt)

| run / scale | p mean \|·\| | p std across images (CV) | pre-tanh std | x std (rad) | ⟨Z⟩ std | group-mean std ÷ member-channel std | random Linear(2C,8) std |
|---|---|---|---|---|---|---|---|
| cbam_rgvqf P3 | 0.116 | 0.0053 (0.08) | 0.0024 | 0.0074 | 0.0079 | 0.45 | 0.0060 |
| cbam_rgvqf P4 | 0.133 | 0.0104 (0.16) | 0.0035 | 0.0108 | 0.0041 | 0.33 | 0.0125 |
| cbam_rgvqf P5 | 0.036 | 0.0069 (0.46) | 0.0017 | 0.0053 | 0.0020 | 0.25 | 0.0086 |
| complete P3 | 0.102 | 0.0174 (0.20) | 0.0128 | 0.0396 | 0.0513 | 0.74 | 0.0198 |
| complete P4 | 0.280 | 0.0415 | 0.0195 | 0.0411 | 0.151 | 0.47 | 0.084 |
| complete P5 | 0.058 | 0.0065 (0.22) | 0.0021 | 0.0065 | 0.0073 | 0.32 | 0.0083 |

**Verdict: the hypothesis is partly confirmed.** It is not the main cause.

- **Pooling does remove variation.** Averaging 32/64-channel groups keeps only 25–47 % of a single channel's across-image std (74 % in `complete` P3).
- **But the dominant cause is scale.** The globally pooled features themselves vary by only ~0.005–0.04 in absolute terms. An *untrained random* `Linear(2C,8)` on the same p gives only ~2–4× the group-mean std, which would still be a near-constant angle (≈0.01 rad).
- **Normalisation is what the encoder needs.** Without a normalising or amplifying step, the angles stay tiny whatever the projection. B6's BatchNorm is therefore the essential part.

## B6-fix — `rgvqf_learned_encoder`: x = π·tanh(BN(Linear(2C→8)(p)))

- **Placement.** REM still weights the RGB/IR halves before the projection. `to_gate` and `spatial_mix` init are unchanged, so the block still starts as additive fusion (`to_gate` = 0).
- **Batch-of-1 safety.** `_BatchNorm1dB1` is `nn.BatchNorm1d` but uses running stats when B=1 in training mode. LLVIP's last train batch per epoch has B=1 (6957 mod 4), where plain BN raises. Verified: a B=1 train-mode forward works.
- **Flags OFF still reproduces HEAD** bit-exactly: 5 configs, state-dict SHA-256 and loss vector.
- **Gradients reach the new layers** once `to_gate` ≠ 0: `encoder.W` 1.1–8.6, `encoder_bn.weight` 0.04–0.06, `q_weights` 4e-4–1.5e-3.

⟨Z⟩ std across 100 val images **at init** (mean per qubit):

| | P3 | P4 | P5 |
|---|---|---|---|
| B6 OFF (eval) | 0.0041 | 0.0046 | 0.0057 |
| B6 ON (eval: BN running stats still at init values 0/1) | 0.0066 | 0.0113 | 0.0143 |
| B6 OFF (train-mode forward) | 0.0023 | 0.0019 | 0.0024 |
| **B6 ON (train-mode BN = batch stats)** | **0.282** | **0.194** | **0.151** |

- **In training** (BN uses batch statistics), the circuit outputs vary 60–120× more than without B6. Angles span ±2.9 rad with only 1–3 % near saturation.
- **In eval mode at init**, the gain is only 2–3×. This is expected: the BN running statistics haven't been calibrated yet. After training they track the real distribution, so eval should approach the train-mode numbers. **The pilot measures this on a trained last.pt.**
- **Caveat.** With batch 4, train-mode BN normalises each image against the other 3 in its batch, which adds noise to x during training. An alternative is LayerNorm over the 8 dims (per sample, no batch coupling).

## B7 — `cbam_rgvqf_classical_ctrl` (in `EXTRA_ABLATIONS`, not `ABLATIONS`)

- **Classical stand-in.** Identical to cbam_rgvqf with B6 ON, but `q_out = tanh(Linear(8,8)(x))`, with **0 QNode calls** per forward (the quantum variant makes 3).
- **No quantum parameters.** `q_weights` is not a parameter (`param_report` quantum = 0). `classical_q` adds 72 params per level, against 32 quantum angles per level.
- **Tightly paired init.** The q_init random draw is still made, and `classical_q` is created under a saved/restored RNG state. As a result, **all 1,030 shared tensors are bit-identical** to cbam_rgvqf (B6 ON) at the same seed. My first version was *not* identical (it advanced the RNG); fixed and re-verified.
- **Training signal.** Forward/backward is finite, and gradients reach `classical_q` (0.55–0.72) and the encoder.

## Also found: the resume path in `run_step5_retrain.train_config`

`YOLO(last).train(resume=True)` without `trainer=` makes Ultralytics fall back to the stock `DetectionTrainer`, which would feed bare RGB to the model.

**No existing run was affected:** 24/24 runs trained fresh, with no resume lines in any log. The B6 pilot driver resumes with `trainer=<custom class>`, and the Part C driver will do the same.
