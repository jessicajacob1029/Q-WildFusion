# Pre-retraining verification — two gates

Device: `mps` (Apple M1 Pro). `ultralytics` 8.3.246, `torch` 2.8.0.
No source files under `phase3/code/` were edited. All instrumentation was runtime
monkeypatch inside throwaway scripts (`diagnostics/task_a_probe.py`,
`diagnostics/task_b_root_delta.py`), both deleted-artifact-free except for their
own script files and the CSV/md outputs listed below. `diagnostics/tmp_probe`
(Task A's 1-epoch scratch run) was deleted after the probe completed.

---

## Task A — Is the fix in the TRAINING-TIME validator?

### Static structure

`RGBTDetectionValidator.preprocess` — `phase3/code/rgbt_trainer.py:31-59`. Calls
`super().preprocess(batch)` (stock device-move/normalize), then reuses
`self._rgbt_trainer._resolve_ir_root()` / `._matching_ir_path()` (the trainer's own
methods, called by reference) to pack `batch["img"]` as an `RGBTBatch` 2-tuple, with
shape/type asserts.

`RGBTDetectionTrainer.get_validator` — `phase3/code/rgbt_trainer.py:97-101`:
```python
def get_validator(self):
    self.loss_names = "box_loss", "cls_loss", "dfl_loss"
    return RGBTDetectionValidator(
        self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks, trainer=self
    )
```
Statically: `get_validator()` returns `RGBTDetectionValidator`, the corrected class —
not stock `DetectionValidator`.

### Runtime probe (decisive)

Ran one real epoch through the unmodified `RGBTDetectionTrainer` (`fraction=0.02`,
`imgsz=320`, `batch=2`, `data=phase3/dataset_aligned/data.yaml`), with both
`RGBTDetectionValidator.preprocess` and stock `DetectionValidator.preprocess`
monkeypatched to log a tag + `self_type` + output shape, and `RGBTDetectionModel.forward`
patched to log input structure whenever `model.training is False` (eval-mode = validation).

Val split = 1740 images, batch=2 → 870 batches expected.

- `preprocess()` calls captured: **1740** = 870 tagged `STOCK(DetectionValidator).preprocess`
  + 870 tagged `CORRECTED(RGBTDetectionValidator).preprocess`, **every one with
  `self_type == 'RGBTDetectionValidator'`**. The "STOCK" tag fires only because
  `RGBTDetectionValidator.preprocess` calls `super().preprocess()` internally — `self`
  is still the corrected subclass instance in both cases. Stock `DetectionValidator`
  was never instantiated/called on its own. 870 = exactly one call per real val batch.
- `RGBTDetectionModel.forward()` eval-mode calls: **873** total — 870 got `RGBTBatch`
  (tuple, shapes `(4,3,288,352)`×2, i.e. real batches — note: actual batch composition
  varied slightly under `rect=True`), 3 got a bare `Tensor` of shape `(1,3,32,32)`.
  These 3 have **no matching preprocess() call** (1740, not 1746) and align exactly with
  the two `"YOLO11n summary: ... GFLOPs"` lines in the log — Ultralytics' internal
  model-profiling routine (`get_flops`/model summary), which calls the model directly
  with a small dummy tensor during trainer setup, bypassing `get_validator()`/
  `preprocess()` entirely. This is unrelated to checkpoint selection or `patience`.

My script's first-pass automated verdict (`all(is_tuple for eval-mode forward calls)`)
flagged this as FAIL — that check was too strict, since it counted the 3 unrelated
profiling calls as if they were validation-dataloader calls. Correcting for that:

### Verdict: **PASS**

All 870/870 real end-of-epoch validation batches received the paired `(RGB, IR)` tuple
via the corrected preprocess. `get_validator()` correctly installs
`RGBTDetectionValidator` as the trainer-time / checkpoint-selection / patience
validator. No code change required.

---

## Task B — Which dataset root is ground truth: `phase3/dataset` or `phase3/dataset_aligned`?

### 1–2. Enumeration and label comparison

| root | rgb/train | rgb/val | ir/train | ir/val | labels |
|---|---|---|---|---|---|
| `phase3/dataset` | 8697 | 2174 | 12025 | 3463 | `labels/train` (8697), `labels/val` (2174) — **no `labels/rgb/`** |
| `phase3/dataset_aligned` | 6957 | 1740 | 6957 | 1740 | `labels/rgb/train` (6959), `labels/rgb/val` (1740) — matches rgb/ir 1:1 |

`phase3/dataset`'s rgb/ir counts don't match each other (raw, unpaired dump); `dataset_aligned`'s do, at both split sizes — consistent with it being a curated, paired subset.

**Val-split filename overlap between the two roots: 0 / 2174.** The two roots use
completely different train/val partitions of an overlapping underlying image pool —
this isn't the same val images with different labels, it's a different split.

Pooling train+val and comparing by filename stem instead: `dataset` has 10871 stems
total, `dataset_aligned` has 8699; **8697 stems are shared, and every one of the 8697
shared label files is byte-identical between the two roots** (0 differ). The other
2174 stems in `dataset` (its entire original val split) are excluded from
`dataset_aligned` entirely — filtered out, not relabeled.

**Conclusion: no label corruption/offset exists anywhere the two roots overlap.** The
only difference is curation (which images are kept and how they're split), not label
content.

### 3. Overlay inspection — not applicable as literally specified

The task asked to overlay both roots' boxes on the *same* val image. Since the two
roots' val splits share **zero filenames**, there is no image to overlay both roots'
boxes onto — the comparison as specified has no target. In its place: the full
byte-identical comparison above (8697/8697 shared-stem labels, not just 6 sampled
images) is strictly stronger evidence of label correctness than 6 manual overlays
would have been.

### 4. Quantify the stake — single checkpoint, both roots

Checkpoint: `runs/detect/abl_no_cbam_add/weights/best.pt`, corrected validation path,
identical settings otherwise.

| root | data.yaml | status | mAP50 | mAP50-95 | wall (s) |
|---|---|---|---|---|---|
| `phase3/data.yaml` (path: `phase3/dataset`) | `phase3/data.yaml` | **FAILED** | — | — | 19.4 |
| `phase3/dataset_aligned/data.yaml` | `phase3/dataset_aligned/data.yaml` | SUCCESS | 0.7206 | 0.2675 | 199.7 |

`phase3/data.yaml` fails immediately with:
```
File "phase3/code/rgbt_trainer.py", line 127, in _resolve_ir_root
    raise KeyError("Dataset yaml must define an 'ir' path, e.g. ir: images/ir")
KeyError: "Dataset yaml must define an 'ir' path, e.g. ir: images/ir"
```
`phase3/data.yaml` has no `ir:`/`thermal:` key at all — it cannot be fed through the
RGBT pipeline, period. It fails loudly, not silently. (`./data.yaml` at the repo root
is a byte-for-byte duplicate of `phase3/data.yaml`, same missing key, same failure.)

The `dataset_aligned` result (mAP50=0.7206, mAP50-95=0.2675) matches the Step 3
re-validation number for this exact checkpoint (mAP50=0.721, mAP50-95=0.268) within
rounding — confirms reproducibility of the corrected path.

**A repo-wide grep confirms `phase3/data.yaml` and root `./data.yaml` are referenced by
zero active scripts.** Every training script (`train.py`, `train_ablation_study.py`,
`train_baseline.py`, `train_90plus.py`, `train_cbam_only.py`,
`train_cbam_vqc_enhanced.py`, `cloud_train_90plus.py`), every eval script
(`eval_area_ap.py`, `eval_filtered_map*.py`, `eval_tta.py`, `run_filtered_val.py`),
every gradcam/qualitative script, `extract_three_models_results.py`, `fix_labels.py`,
and all diagnostics scripts hardcode or pass `phase3/dataset_aligned/...` — and every
`args.yaml` in all 10 ablation-grid runs records `data: .../phase3/dataset_aligned/data.yaml`.

### Verdict

**Ground truth root = `phase3/dataset_aligned`.** Justification: (a) it is the only
root the RGBT pipeline can even run against — `phase3/data.yaml`/`./data.yaml` crash
immediately on a missing `ir:` key; (b) it is the sole root referenced by every
training, evaluation, gradcam, and diagnostic script in the repo, with zero exceptions
found; (c) where the two roots' image pools overlap, labels are byte-identical — no
corruption exists to be masked.

Critically: **this was already the root used for every ablation-grid training run and
for the Step 3 re-validation** — `dataset_aligned` was never swapped in as a fix, it
was already what the pipeline used throughout. The `phase3/dataset` / top-level
`data.yaml` files are dead, orphaned artifacts (likely leftover from an earlier,
unaligned/raw-LLVIP stage) that happen to sit in the repo but were never live.

No `diagnostics/data_aligned.yaml` was created — unnecessary, since
`phase3/dataset_aligned/data.yaml` already exists and is already what's live; nothing
needs promoting.

---

## Deliverables produced
- `diagnostics/step4b_verification.md` (this file)
- `diagnostics/task_a_probe.py` — Task A instrumentation script
- `diagnostics/task_b_root_delta.py` — Task B step-4 script
- `diagnostics/task_b_root_delta.csv` — the both-roots single-checkpoint delta
- No overlay jpgs (not applicable — see Task B §3)
- No `diagnostics/data_aligned.yaml` (not needed — see verdict)
- `diagnostics/tmp_probe/` deleted after use

## Stop / hand-back

**Task A: PASS. Task B: ground truth = `dataset_aligned`, which is what training and
Step 3/4 already used.**

Both gates are clear. The existing Step 3/4 numbers stand as-is — no re-run of any
kind is needed, since the ablation grid was already trained and already
re-validated against the correct root. Nothing further to do before spending
120-epoch compute on this front; that decision is independent of these two gates.
