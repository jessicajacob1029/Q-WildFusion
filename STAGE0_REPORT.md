# Stage 0 Reconnaissance Report

Scope: read-only investigation. No module or training code was written or
modified. All numbers below were produced by actually running snippets
against the live repo (model instantiation, image sampling, label scanning,
environment checks) — nothing is assumed from the plan doc without checking.

---

## 1. `vqc_fusion.py` and `cbam.py` — confirmed behavior

**`phase3/code/model/vqc_fusion.py` (`VQCFusion`) is classical, not quantum.**
No PennyLane import anywhere in the file. The forward pass:
1. GAP each modality → `Linear(channels*2, latent_dim=8)` → `q`
2. `q = sin(q·theta) + cos(q·phi)` — elementwise trig, `theta`/`phi` are plain
   `nn.Parameter` vectors, not circuit gate angles.
3. "Entanglement-like mixing" = `q + roll(q, shift=1)`, a fixed shift-and-add,
   not any operator acting on a Hilbert space.
4. `Linear(latent_dim, channels)` → sigmoid → channel gate, applied to a
   learned 1×1-conv spatial mix of `[fr, ft]`.

This confirms the plan's claim exactly: it's a sin/cos-parameterized gating
MLP with cosmetic "quantum-inspired" naming, wired in as the default
`fusion_mode="vqc"` in `rgbt_backbone_cbam.py`.

**`cbam.py`** is a textbook CBAM (channel attention via avg+max pool → shared
MLP → sigmoid, then spatial attention via channel-avg+max → 7×7 conv →
sigmoid), instantiated **twice per pyramid level, once per modality**
(`rgb_cbam_p3/p4/p5`, `ir_cbam_p3/p4/p5` in `rgbt_backbone_cbam.py`). Each
CBAM only ever sees its own modality's tensor — there is no cross-modal
conditioning, confirming "CBAM is off-the-shelf, per modality" from the plan.

**Real tensor shapes** (measured by instantiating `RGBTDetectionModel` and
running `fusion_backbone.forward` on dummy `[1,3,64,64]` RGB/IR tensors — see
§2 for the harness):

| Scale | P3 | P4 | P5 |
|---|---|---|---|
| yolo11n (`train.py`'s active model) | `[B,128,H/8,W/8]` | `[B,128,H/16,W/16]` | `[B,256,H/32,W/32]` |
| yolo11x (`baseline_train.py`, `train_cbam_only.py`, `train_cbam_vqc_enhanced.py`, `train_90plus.py`, `cloud_train_90plus.py`) | `[B,768,H/8,W/8]` | `[B,768,H/16,W/16]` | `[B,768,H/32,W/32]` |

Note P3==P4==P5==768 for the x-scale — that's not a bug, it's `max_channels=512`
in the yolo11 yaml `scales:` table capping the pre-multiplier channel count for
those three backbone stages identically before the 1.5× width multiplier is
applied.

---

## 2. Layer-index "bug" — investigated, **not reproducible as stated**

The plan hypothesized that `predict_rgbt`'s hard-coded neck indices `11–23`
assume `yolo11s` and would misalign for `yolo11x`/`yolo11n`. I tested this
directly rather than trusting the hypothesis:

```
YOLO("yolo11n.pt") → cfg.scale = 'n' → RGBTDetectionModel → predict_rgbt(rgb, ir) → OK
YOLO("yolo11x.pt") → cfg.scale = 'x' → RGBTDetectionModel → predict_rgbt(rgb, ir) → OK
```

Both ran end-to-end with no shape or index errors. Root cause of why the
worry doesn't hold: Ultralytics' `scales:` mechanism (`n/s/m/l/x` in
`yolo11.yaml`) only multiplies **channel widths** (`width_multiple`) and
**block repeat counts** (`depth_multiple`, applied *inside* a `C3k2`/`C2PSA`
module, not as new top-level graph nodes) and caps via `max_channels`. It does
**not** change the number or order of top-level entries in the YAML backbone/
head list. I diffed the installed `ultralytics` package's
`cfg/models/11/yolo11.yaml` against `rgbt_backbone_cbam.py`'s taps and
`yolov11_rgbt.py`'s hand-unrolled neck:

- Taps at backbone indices `4` (P3, after the first P3-stage `C3k2`), `6` (P4,
  after the P4-stage `C3k2`), `10` (P5, the `C2PSA` output) — match the yaml's
  own `# cat backbone P3` / `# cat backbone P4` / `# cat head P5` comments
  exactly.
- `predict_rgbt`'s manual replay of layers `11`→`23` (Upsample→Concat→C3k2 ×2,
  then downsample→Concat→C3k2 ×2, then `Detect([16,19,22])`) is index-for-index
  identical to the yaml head, for every scale, because `self.model` is built
  by `parse_model()` from that same yaml regardless of which `.pt` is loaded.

**So: no fix needed for the scale question as posed.** The real fragility is
different and narrower: `predict_rgbt` assumes the *stock* `yolo11*.yaml`
detect-head topology (24 top-level modules, `Detect` at index 23, taps at
4/6/10). It would silently break only if someone pointed `cfg` at a
non-standard yaml (e.g. `yolo11-p6.yaml`, a custom neck, or a future
Ultralytics head redesign) — worth a documented assumption/assertion, not a
scale-conditional index table. I'd recommend adding a cheap runtime check
(`assert isinstance(layers[23], Detect) and taps come from a C3k2/C2PSA`)
rather than deriving indices dynamically — the current code is correct, just
un-guarded against a topology it doesn't support.

**Second finding, more concrete:** `RGBTDetectionModel.__init__`'s default
`cfg="yolo11s.yaml"` is dead — every real call site (`rgbt_trainer.py`
`get_model`) passes `cfg` explicitly, sourced from the `.pt`'s own embedded
yaml, so the default is never hit in practice. Not a bug, just an unused/
misleading default worth removing or fixing when this file is touched.

---

## 3. Origin of `quantum_features/features_val.npy`

Written by `phase3/code/extract_features.py`. Key facts, all read directly
from the file:

- Loads `frozen_models/qwildfusion_classical_best.pt` (exists, 22.5 MB) via
  `YOLO(weights_path).model`, frozen (`requires_grad=False`).
- **RGB-only.** It iterates `phase3/dataset_aligned/images/rgb/val` — thermal
  images are never loaded in this script. The "quantum fusion" trained on
  these features has never seen a thermal pixel.
- Runs `model.model[:10]` (backbone only, single-stream) → GAP → `reduce_dim`.
- **`reduce_dim` instantiates a brand-new `torch.nn.Linear(C, 6)` on every
  call**, with default (untrained, random) init — it is never loaded from a
  checkpoint or fit to anything. So `features_val.npy` is a frozen backbone's
  GAP output passed through an **untrained random projection**, not a learned
  descriptor.
- Output: 1740 rows (matches val split count) × 6 dims, saved alongside
  `filenames_val.npy`.

Downstream, `quantum_fusion.py` defines the real PennyLane circuit
(`default.qubit`, 6 qubits, 2 variational layers, ring-CNOT entanglement,
`diff_method="parameter-shift"`, per-sample Python loop over the batch — no
batching/broadcasting). `train_quantum_fusion.py` trains it for 50 epochs on
that static `.npy` file using only `variance_loss` (negative mean feature
variance) + `cosine_similarity_loss` (mean pairwise cosine sim, encourages
decorrelation) — a **self-supervised proxy loss with no detection signal**,
confirming the plan's "contributes zero to mAP" framing precisely, and adding
the previously-undocumented detail that it's also RGB-only and driven by an
untrained random linear projection rather than a learned one.
`quantum_fusion_no_entangle.py` mirrors this exactly minus the CNOT ring —
already a ready-made ablation control for the "no-entangle" row in Part B.

---

## 4. `data.yaml` / IR path resolution

`phase3/dataset_aligned/data.yaml`:
```yaml
path: /Users/test/Q-WildFusion/phase3/dataset_aligned
train: images/rgb/train
val: images/rgb/val
ir: images/ir
nc: 1
channels: 3
names: [human]
```
`ir:` is present and resolves via `RGBTDetectionTrainer._resolve_ir_root` +
`_matching_ir_path` (join `ir_root/<split>/<basename>`), matching the actual
on-disk layout `images/ir/train/` and `images/ir/val/`. Verified counts:

| Split | RGB images | IR images |
|---|---|---|
| train | 6957 | 6957 |
| val | 1740 | 1740 |

Counts match 1:1 per split (matches the plan's 6957/1740 figures), and every
RGB file has a same-named IR counterpart under the corresponding split
directory.

---

## 5. `runs/detect/` inventory vs. `train_ablation_study.py`

`ABLATIONS` currently defines only **3** entries, not the full Part B grid
(expected — CMAAM/REM/RGVQF/MQFP/composite-loss rows don't exist yet):

| key | run name | fusion_mode | use_cbam |
|---|---|---|---|
| `no_cbam_add` | `abl_no_cbam_add` | add | False |
| `cbam_add` | `abl_cbam_no_vqc` | add | True |
| `cbam_vqc` | `abl_cbam_vqc_proposed` | vqc | True |

Cross-checking `runs/detect/` against these names:

- `abl_no_cbam_add` exists on disk with matching `args.yaml` (`model:
  yolo11n.pt`, matches script default) but **no `results.csv`** — the run
  didn't complete or wasn't validated.
- `abl_cbam_no_vqc` / `abl_cbam_vqc_proposed` (current script's expected
  names) are **not present** on disk at all.
- `abl_add`, `abl_concat`, `abl_vqc_real` exist but their `args.yaml` shows
  `name: yolo11_cbam_rgbt3` / `yolo11_cbam_rgbt4` internally and `model:
  yolo11s.yaml` — these are renamed/relocated outputs from an **earlier,
  different version** of the training script (yolo11s, not the current
  yolo11n default), predating the current `ABLATIONS` dict. `abl_concat`'s
  fusion mode ("concat") isn't even a key in the current `ABLATIONS` dict —
  concat has never been run through `train_ablation_study.py` as it exists
  today.
- `ablation_pilot/` is empty (no `args.yaml`, no `results.csv`).
- The rest of `runs/detect/` (`train`, `train2`, `sanity_rgb_ir*`,
  `rgb_ir_cbam_final*`, `rgb_thermal_cbam_final*`) load `yolov8n.pt` /
  `yolov8s.yaml` — remnants of the dead YOLOv8 prototype track, unrelated to
  the current YOLOv11 pipeline.
- `yolov11X_baseline`, `yolov11X_vqc_cbam_90plus_target(2)` load `yolo11x.pt`
  and correspond to `baseline_train.py` / `train_90plus.py` /
  `cloud_train_90plus.py`, not to `train_ablation_study.py`.

**Net finding:** there is no clean, current on-disk ablation grid to build
on. `abl_no_cbam_add` is the only run whose name and config match the live
script, and even it lacks results. Part B's grid will need fresh runs.

**Additional dead-code finding (not asked for, but blocking if reused):**
`phase3/code/model/RGBIR_YOLOv11_CBAM.py` and
`phase3/code/model/rgbt_fusion_model.py` are alternate, unused model
implementations (single-CBAM-tap, add-only fusion; a `RGBThermalFusion`
wrapper class respectively) — neither is imported by `rgbt_trainer.py` or any
active `train_*.py`. `phase3/code/predict.py` imports
`RGBIR_YOLOv8_CBAM` from `phase3.code.model`, which doesn't exist in
`model/__init__.py` (empty file) or under any filename in `model/` — that
import is currently broken. None of this blocks Stage 1–2 since the live
path is `yolov11_rgbt.py` → `rgbt_backbone_cbam.py` → `vqc_fusion.py`/`cbam.py`,
but it's worth flagging so these aren't mistaken for alternatives to extend.

---

## 6. Environment

```
pennylane      0.38.0
lightning.qubit  available (qml.device("lightning.qubit", wires=4) succeeds)
torch          2.8.0
CUDA available: False
MPS available:  True
```

Device handling today: the **only** place a QNode currently runs
(`quantum_fusion.py` / `quantum_fusion_no_entangle.py`, via
`train_quantum_fusion.py`) hardcodes `DEVICE = "cpu"` and never touches
MPS/CUDA — it sidesteps the issue entirely because it's an offline script
over a static `.npy` file. There is **no existing code path that mixes a
PennyLane QNode with an MPS tensor**, so Stage 2 (RGVQF running in-detector,
where the rest of the model trains on `device="mps"` per `train.py`) will be
the first time this integration is actually exercised. Concretely this means:
descriptor tensors feeding the QNode will need an explicit `.cpu()` hop
before `quantum_circuit(...)` and `.to(device)` after, since PennyLane's
`default.qubit`/`lightning.qubit` simulators run on CPU only — no native MPS
backend exists for either. `diff_method="parameter-shift"` is also worth
reconsidering per the plan's own note: `lightning.qubit` + `torch` supports
`diff_method="adjoint"` (exact, faster than parameter-shift, doesn't need the
per-sample Python loop to be eliminated to help) — swapping `default.qubit`
→ `lightning.qubit` and `parameter-shift` → `adjoint` is a lower-risk first
optimization than chasing `backprop`, which `lightning.qubit` doesn't support
for this circuit shape.

---

## 7. LLVIP class count / night-only

Confirmed empirically, not just from `data.yaml`:

- **`nc=1`, single class.** Scanned class IDs across the first 2000 label
  files in `labels/rgb/train/`: only `{'0'}` appears. Matches `data.yaml`'s
  `names: [human]`.
- **Night-only.** Sampled 8 random RGB train images and measured mean
  grayscale brightness (0–255 scale): values ranged **20.8–64.7**, all
  consistent with low-light/night capture (a well-lit daytime frame would
  typically sit >100). This corroborates the plan's LLVIP night-only premise
  with direct pixel evidence rather than taking it on faith.

This locks in the Dataset Limitations section as written in the plan: no
Day/Fog/Rain rows are derivable, and REM's day→RGB/night→thermal story has no
support in this data — the thermal-contrast reframing is the only honest
option without adding a second dataset.

---

## Surprises / things not in the original plan

1. **`features_val.npy` is RGB-only and passes through an untrained random
   linear projection**, not a learned one. The offline quantum circuit has
   literally never seen a thermal pixel or a trained projection — worse than
   "zero contribution to mAP," it's disconnected from both modalities'
   trained representations.
2. **The layer-index "bug" doesn't reproduce.** Verified directly on both
   yolo11n and yolo11x — Ultralytics' scale mechanism is width/depth-only, no
   graph restructuring. Recommend dropping that item from the Stage 1 task
   list and replacing it with a lightweight topology assertion instead.
3. **No usable current ablation grid on disk** — only one of three expected
   run directories exists, and it has no results. Part B will start from
   zero, not from partially-existing baselines.
4. **`predict.py` has a broken import** (`RGBIR_YOLOv8_CBAM` doesn't exist)
   and two other model files (`RGBIR_YOLOv11_CBAM.py`, `rgbt_fusion_model.py`)
   are unused prototypes — not blocking, but flagged so they aren't mistaken
   for extension points.
5. **`RGBTDetectionModel`'s default `cfg="yolo11s.yaml"` is dead code** — every
   real trainer call overrides it via the loaded checkpoint's own yaml.

## Recommended approach for Stage 1–2

- Stage 1 (fusion registry): proceed as planned — extracting
  `concat_pX`/`vqc_pX`/`if/elif` into `build_fusion(mode, ch)` is a
  mechanical refactor with no blockers found here. Add the topology assertion
  from §2 while touching `yolov11_rgbt.py`, as a cheap guard, not a
  scale-indexed lookup table.
- Stage 2 (RGVQF): do **not** reuse `features_val.npy` or its extraction
  pipeline as-is (finding #1) — the new descriptor must be built from live
  RGB+IR P-level features inside the forward pass, weighted by REM, not from
  the offline single-modality artifact. Start the PennyLane integration with
  `lightning.qubit` + `diff_method="adjoint"` rather than
  `default.qubit`/`parameter-shift`, given both are confirmed available and
  `adjoint` avoids the shot-multiplying cost of parameter-shift outright. Budget
  first for a CPU round-trip on the descriptor tensor (§6) before optimizing
  batching, and benchmark one training step at real image size before
  committing to a full run, per the plan's own guardrail.
- Ablation grid (Part B): treat as a fresh start, not a resume — none of the
  three expected baseline runs are complete and usable.

Stopping here per instructions — awaiting go-ahead before any Stage 1 code.
