# Step 0 — cost de-risking and gate (Q-WildFusion staged retrain)

Device: `mps` (Apple M1 Pro). `ultralytics` 8.3.246, `torch` 2.8.0. Live data config
confirmed: `phase3/dataset_aligned/data.yaml` (has `ir:` key), used by both
`train.py` and `train_ablation_study.py`. Kickoff assertion passed: `get_validator()`
source references `RGBTDetectionValidator`.

## Step 0a — diff_method / QNode audit

Three PennyLane QNodes exist in the repo:

| file | device | diff_method | reachable from the 10-config grid? |
|---|---|---|---|
| `phase3/code/quantum_fusion.py` | `default.qubit` | `parameter-shift` | **No** — `FUSION_REGISTRY["vqc"]` maps to `QInspiredFusion` (`model/fusion/qinspired.py`), a pure classical sin/cos-gated module with zero PennyLane calls. This file is dead code w.r.t. the trainer. |
| `phase3/code/quantum_fusion_no_entangle.py` | `default.qubit` | `parameter-shift` | **No** — only imported by the standalone `run_quantum_fusion_no_entangle.py` script, not the trainer/fusion registry. |
| `phase3/code/model/fusion/rgvqf.py::_rgvqf_circuit` | `lightning.qubit` | `adjoint` | **Yes** — the only QNode any of the 10 grid configs can reach (6/10 configs use `fusion_mode="rgvqf"`). |

`rgvqf.py` is already on the cheapest correct configuration (adjoint on a
state-vector simulator = exact gradient, fastest available). Its own docstring
documents the exact benchmark that motivated this: the naive per-sample
parameter-shift loop (the offline `quantum_fusion.py` approach) measured ~4.9s for
a 3-scale batch-4 step; the current batched `lightning.qubit`+`adjoint` circuit
measures ~0.04s for the same shape — already a ~120x improvement, already shipped.

`abl_complete`'s `beta_q=0.1` activates `quantum_regularization_loss` (L_q) —
confirmed a pure classical function (variance + pairwise cosine similarity) over
`RGVQFFusion.last_q_out`, the tensor already produced by the adjoint circuit. It
does not invoke any additional QNode.

**No diff_method switch was made — none is applicable.** There is no unused lever
here; the live quantum path is already optimal.

## Step 0b — timing probe

One real, full epoch (imgsz=960, batch=4, full train+val split, corrected validator
end-of-epoch — the actual grid settings, not the CLI script's defaults) on
`no_cbam_add` (fast/classical) and `cbam_rgvqf` (quantum):

| config | fusion | train_epoch | in-loop val | one-time overhead* | total wall |
|---|---|---|---|---|---|
| `abl_no_cbam_add` | add (classical) | 1212.5s | 423.3s | 967.6s | 2603.3s |
| `abl_cbam_rgvqf_proposed` | rgvqf (quantum) | 1305.7s | 169.2s | 370.3s | 1845.2s |

\* one-time overhead = dataset-cache scan (negligible, ~0.0s once cached) + a
**second, separate validation pass** that Ultralytics' `BaseTrainer.final_eval()`
runs after training completes, via `self.validator(model=best.pt_path)` — a
different code path (`trainer=None`, routes through `AutoBackend`) than the one
Task A verified (`trainer=self`, raw model). This was not previously checked; I
verified it here: it is **not broken** — both runs produced sane, non-degenerate
P/R (`no_cbam_add`: in-loop P=0.759/R=0.506/mAP50=0.542 vs final_eval
P=0.766/R=0.600/mAP50=0.646 — the gap is the standard EMA-vs-saved-checkpoint
difference, not a bug). This overhead is paid once per training run, not once per
epoch, so it does not scale with epoch count — noted for completeness, not a cost
driver at the epoch counts this task budgets for.

- **Train-epoch ratio, quantum:fast = 1 : 1.08** — RGVQF adds only ~8% to raw
  training time. The quantum lever is essentially free at this point.
- The in-loop val-time difference (423.3s vs 169.2s) is very likely NMS/box-count
  noise specific to epoch-1 (heavily undertrained, differing spurious-detection
  rates between architectures) rather than a stable architectural effect — flagged,
  not treated as signal.

## Step 0c — Stage-1 grid cost extrapolation

10 configs = 4 fast (`no_cbam_add`, `cbam_no_vqc`, `cbam_vqc_proposed`, `cmaam_add`)
+ 6 quantum (`cbam_rgvqf_proposed`, `cbam_rgvqf_no_entangle`, `cbam_rgvqf_rem`,
`cmaam_rgvqf_rem`, `mqfp_single_scale`, `complete`). Per-epoch cost = train_epoch +
in-loop val only (the one-time overhead is added once per config, not per epoch).
**No early-stopping credit is taken** — this is the full-budget ceiling, since
patience behavior under the corrected signal is genuinely unknown (could fire
earlier *or* later than the old grid's noise-driven early stops; treated as
uncertain rather than guessed).

| epochs/config | Σ per-epoch (all 10) | + one-time overhead ×10 | **total** |
|---|---|---|---|
| 50 | 769,630s | 9,676s | **216.5 hours (9.0 days)** |
| 80 | 1,231,408s | 9,676s | **344.7 hours (14.4 days)** |
| 120 | 1,847,112s | 9,676s | **515.8 hours (21.5 days)** |

Sequential, single MPS device — matches the original grid's real multi-day runtime
at 35 epochs/config, so this isn't a surprising number, just a large one.

## Step 0d — GATE

`MAX_GRID_HOURS = 36`. Extrapolated Stage-1 cost at `MODERATE_EPOCHS=50` = **216.5
hours ≈ 6.0x over budget**. Even the most aggressive reasonable epoch count that
would fit 36h (running all 10 configs, no early-stop credit) is **~7.8
epochs/config** — far short of anything that would let `patience=20` mean
anything, and far short of what the original grid needed (18–35 epochs) to reach
even its noisy plateau.

### GATE: FAIL — stopping here. Not proceeding to Step 1. No training launched.

## Options handed back (per the gate-failure protocol)

| option | effect | note |
|---|---|---|
| **diff_method switch** | none available | Step 0a already confirmed the live QNode is at the cheapest correct configuration; this lever is exhausted, not just untried. |
| **Lower epochs** | ~7.8 epochs/config fits 36h for all 10 configs | Almost certainly too few to reach any real plateau (original grid needed 18–35); ranking at this budget would likely reproduce the "undertrained, still-climbing" problem Step 2's own flagging step exists to catch — not recommended as-is. |
| **Train a subset of configs** | e.g. 3 core configs (`no_cbam_add`, `cbam_rgvqf_proposed`, `complete`) at 50 epochs ≈ **64.5h** — still over 36h but far closer; 5 configs ≈ 108.2h | Narrows the ablation (loses some pairwise comparisons the task cares about — e.g. entangle vs no-entangle, REM on/off) but is the most direct lever left. |
| **Raise `MAX_GRID_HOURS`** | accept the real cost: ~217h (9.0 days) for all 10 configs at 50 epochs, sequential on this Mac | The honest number if the full grid at a real epoch count is non-negotiable. |
| **Rent GPU** | a CUDA card (even a modest one) is typically 5–20x faster than this M1 Pro's MPS backend for this workload | Could plausibly bring the 50-epoch/10-config grid from ~217h down into the 11–43h range, i.e. into or near budget — the only option that doesn't trade away epochs or configs. |

No training was launched past this point, per the guardrail. Awaiting a decision on
which option(s) to take before any Stage-1 run begins.
