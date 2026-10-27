# Pre-flight Parts A + B — report

All Part A checks are read-only on existing checkpoints; raw outputs are `a*.json` / `a*.log` in this folder.
The Part B diff is `partB.diff`; verification outputs are `partB_*.json` / `partB_*.log`.

## Part A — existing checkpoints

### A1. Did the quantum weights move? Yes, at every scale. (`a1_qweights.json`)

Initial `q_weights` were re-created by replaying `run_step5_retrain.train_config`'s construction order with the run's seed (`YOLO()` → trainer init → `init_seeds(seed)` → `get_model`).
The replay is validated: trained weights correlate with the correct-seed init (mean cosine +0.37 / +0.19 / +0.15 at P3/P4/P5) but not with a wrong-seed init (−0.00 / +0.01 / −0.06).

Ranges over 13 RGVQF runs × {best, last}; init ‖q‖ ≈ 0.05:

| scale | ‖Δq‖₂ | max \|Δq\| (rad) | ‖to_gate W‖ (init 0) |
|---|---|---|---|
| P3 | 0.86 – 3.18 | 0.19 – 1.57 | 0.55 – 4.04 |
| P4 | 0.54 – 2.23 | 0.14 – 1.57 | 0.43 – 3.03 |
| P5 | 0.07 – 0.52 | 0.03 – 0.37 | 0.55 – 6.86 |

- P5 moves least (~2–10× its init norm), but nowhere is the change ≈0.
- In `abl_complete`, the P3/P4 max change is exactly ≈π/2 (1.570 rad), consistent with L_q's variance term pushing some RX angles to π/2.

### A2. Does the circuit output vary across images? Barely, in most configs. (`a2_qout_{grid,seeds}.json`, 100 val images)

| config (checkpoint) | per-qubit ⟨Z⟩ std across images (mean, P3/P4/P5) | gate std across images (mean over channels, max channel) |
|---|---|---|
| cbam_rgvqf_proposed, 8 seeds | 0.002–0.009 | 0.0002–0.0015 (max ≤0.003) |
| cbam_rgvqf_no_entangle s0 best | 0.0005–0.0016 | **0.0000** (max ≤0.0001) |
| mqfp_single_scale s0 last (P5) | 0.0004 | 0.0001 |
| cbam_rgvqf_rem s0 | 0.02–0.06 | 0.003–0.007 (max 0.02) |
| cmaam_rgvqf_rem s0 | P3/P4 0.02–0.10, P5 0.002–0.007 | up to 0.013 (max 0.039) |
| complete s0 | P3 0.05, P4 0.15, P5 0.007 | P3/P4 0.008–0.012 (max 0.041), P5 0.001 |

⟨Z⟩ lies in [−1, 1]. The gate mean is 0.24–0.50.

- **Without REM, the quantum channel gate is in effect an image-independent constant per-channel scale.** RGVQF then reduces to `spatial_mix(cat[r,t]) × const_c`.
- **With REM**, it varies, but by at most about ±0.04 on a gate around 0.3–0.5.
- **Likely cause (UNVERIFIED, not measured):** the descriptor is a global average over all H·W positions and 32/64-channel groups, so it barely changes between images; REM's per-image weights are what add variation.
- **This is a method-design issue, not a bug.** Part B does not change it.

### A3. P3 box head zero gradients: benign. (`a3_p3_head.json`, `abl_complete` last.pt, 50 train batches)

- 35 of 50 batches have all-zero gradient on `model.23.cv2.0.*`.
- **Every one** of those 35 has 0 positive anchors assigned to P3, and every batch with P3 positives has non-zero gradient.
- The P3 classification branch (`cv3.0`) gets gradient in 50 of 50 batches.
- Mean positive anchors per batch: P3 1.66, P4 90.6, P5 26.7. LLVIP pedestrians are mostly large: the minimum box height per batch was ≥86 px in the zero-gradient batches. TAL therefore assigns them to P4/P5.
- This is expected behaviour of the box loss (it is computed on positives only). Worth one sentence in the paper: the P3 box branch is rarely used on LLVIP.

### A4. Seeds: the data order is identical across seeds. (`a4_seeds.json`)

| | seed 0 | seed 1 |
|---|---|---|
| first 5 batches (files) | identical | identical |
| batch-1 pixels (HSV augmentation) | differ | differ |
| initial loss `[box,cls,dfl]·B` | 7.280, 10.399, 5.348 | 7.630, 10.605, 5.417 |
| cls-head / CBAM W1 init | differ | differ |

`ultralytics/data/build.py:324-325` seeds the DataLoader generator with a constant (`6148914691236517205 + RANK`), not `args.seed`. "Seeds" therefore vary only weight init and augmentation randomness, never data order. The existing n=8 seed spread probably *underestimates* true run-to-run variance.

### A5. Dead ReLU over the full val set (1,740 images, last.pt). (`a5_relu_*.json`)

| checkpoint | module | active fraction per level (units ever active / total) |
|---|---|---|
| abl_complete | CMAAM rgb / ir | P3 0/32, 0/32 · P4 0/32, 0/32 · P5 0/64, 0/64 — **0.0 everywhere** |
| abl_cmaam_add | CMAAM rgb / ir | P3 0, 0 · P4 0, 0 · P5 rgb 0.016 (1/64), ir 0 |
| abl_cbam_no_vqc | CBAM rgb / ir | P3 0.375 (6/8) / 0.055 (1/8) · P4 0.035 (2/8) / 0.028 (1/8) · P5 0.399 (11/16) / 0.185 (9/16) |

**CMAAM's channel attention is completely dead.** CBAM is mostly, not entirely, dead; the earlier 1-batch audit overstated CBAM death.

### A6. Why the channel ReLU dies (`a6_relu_death.json`, `abl_complete`)

1. **At init, W2 = 0, so the gradient into the hidden layer is W2ᵀg = 0.** W1 gets exactly zero gradient (measured 0.0 at all 6 MLPs), while W2 gets gradient (0.33–2.5).
2. **The channel MLP has no bias**, and its input descriptor (pooled post-SiLU features) is mostly positive and strongly correlated across channels (46–70 % of entries > 0). A row of W1 whose sum drifts negative pushes its unit negative for *every* image at once.
3. **Trained W1 drifted:** mean row-sum went from −0.13…+0.10 (init) to −1.0…−2.0 (trained) at P3/P4. With the init W1, 42–56 % of pre-activations on the same inputs would be positive; with the trained W1 it is 0 % (max pre-activation −0.56 to −1.30).
4. **Once a unit is dead, ReLU passes no gradient, so it can never recover.** W2 then also gets zero gradient (Probe C), and the gate is fixed at 2σ(0) = 1.

## Part B — fixes (all behind flags, default OFF)

| flag (trainer class attribute) | what it does | files |
|---|---|---|
| `attn_channel_fix` (B1) | channel MLP: `LeakyReLU(0.1)` instead of ReLU; W2 ~ N(0, 5e-5) instead of 0 | cbam.py, cmaam.py, rgbt_backbone_cbam.py, yolov11_rgbt.py, rgbt_trainer.py |
| `aux_loss_batch_scale` (B2) | `[α·L_rel, β·L_q] × B`, matching `L_det × B` | yolov11_rgbt.py, rgbt_trainer.py |
| `rem_fused_levels_only` (B3) | build REM only at levels whose fusion is `rgvqf` | rgbt_backbone_cbam.py, yolov11_rgbt.py, rgbt_trainer.py |
| B4 (always on; logging only) | `RGBTDetectionModel.param_report()`, printed by `get_model` | yolov11_rgbt.py, rgbt_trainer.py |

**Backward compatibility.** Forward methods are unchanged and the loss uses `getattr`, so all existing checkpoints still load and evaluate.

### Verification

| check | result |
|---|---|
| Flags OFF reproduces HEAD | Code exported with `git show`; same seed; 4 configs (`cbam_add`, `cmaam_add`, single-scale, `complete`): **state-dict SHA-256 identical and loss vector identical** — PASS (`partB_repro.log`) |
| B1 step-0 gates = 1.0 within 1e-3 | Off: exactly 0. On with std 1e-3 (as proposed): **FAIL**, up to 9.1e-3. On with std **5e-5**: max 2.2e-4 (CBAM), 4.6e-4 (CMAAM) — PASS (`partB_gates.log`) |
| B1 short run (fraction 0.05, 87 batches, 32 optimizer steps, MPS) | After the last step, every level of `cbam_add` (6 MLPs) and `cmaam_add` (6 MLPs): positive pre-activation fraction 0.25–0.75; W1 grad 2e-3–3e-2; W2 grad 0.12–1.15 — PASS (`partB_train_*.log`) |
| B1 recoverability | Dead `abl_complete` last.pt with ReLU → LeakyReLU: W1 gradient goes from 0 to 3e-3–1.5e-1 at all 6 MLPs (`partB_recover.log`) |
| B2 loss vector, same batch | Off: `[6.01062, 10.94121, 4.54726, 0.00784, 0.00997]`. On: `[6.01062, 10.94121, 4.54726, 0.03137, 0.03987]`. Aux ratio exactly 4.0 = B; detection terms identical — PASS |
| B3 single-scale | Off: REM at p3, p4, p5 (used 4,336,137). On: REM at p5 only (used 4,336,117, −20 dead params) — PASS |
| B4 | `param_report()` matches the audit's probe counts for all 10 configs — PASS |

**Limitation of the B1 short run.** 32 optimizer steps cannot reproduce the long-run drift that killed the ReLUs. The guarantee is structural: LeakyReLU has non-zero gradient everywhere, and the recoverability test shows it revives even fully dead units.
