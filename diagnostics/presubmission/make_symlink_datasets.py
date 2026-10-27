#!/usr/bin/env python3
"""
Build per-file symlink datasets under diagnostics/presubmission/data/ so that no
run ever writes into phase3/dataset_aligned. Ultralytics' check_det_dataset
.resolve()s the train/val directories (ultralytics/data/utils.py:458), so the
directories here are REAL directories containing per-file symlinks; label caches
are then written next to these directories, never next to the originals.

  data/fusion/        images/rgb/{train,val} -> RGB files, labels/rgb/{train,val} -> labels/rgb files,
                      data.yaml with ir: <absolute original images/ir> (read-only)
  data/baseline_rgb/  images/{train,val} -> RGB files, labels/{train,val} -> labels/rgb files
  data/baseline_ir/   images/{train,val} -> IR files,  labels/{train,val} -> labels/rgb files

Idempotent. Never writes to the originals.
    .venv/bin/python -I diagnostics/presubmission/make_symlink_datasets.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(ROOT, "phase3", "dataset_aligned")
DST = os.path.join(ROOT, "diagnostics", "presubmission", "data")


def link_dir(src_dir, dst_dir, exts):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src_dir)):
        if not f.lower().endswith(exts):
            continue
        d = os.path.join(dst_dir, f)
        if not os.path.lexists(d):
            os.symlink(os.path.join(src_dir, f), d)
        n += 1
    return n


def main():
    counts = {}
    for split in ("train", "val"):
        counts[f"fusion/images/rgb/{split}"] = link_dir(f"{SRC}/images/rgb/{split}", f"{DST}/fusion/images/rgb/{split}", (".jpg",))
        counts[f"fusion/labels/rgb/{split}"] = link_dir(f"{SRC}/labels/rgb/{split}", f"{DST}/fusion/labels/rgb/{split}", (".txt",))
        for name, img in (("baseline_rgb", "rgb"), ("baseline_ir", "ir")):
            counts[f"{name}/images/{split}"] = link_dir(f"{SRC}/images/{img}/{split}", f"{DST}/{name}/images/{split}", (".jpg",))
            counts[f"{name}/labels/{split}"] = link_dir(f"{SRC}/labels/rgb/{split}", f"{DST}/{name}/labels/{split}", (".txt",))
    with open(f"{DST}/fusion/data.yaml", "w") as f:
        f.write(f"path: {DST}/fusion\ntrain: images/rgb/train\nval: images/rgb/val\nir: {SRC}/images/ir\n"
                "nc: 1\nchannels: 3\nnames:\n  - human\n")
    for name in ("baseline_rgb", "baseline_ir"):
        with open(f"{DST}/{name}/data.yaml", "w") as f:
            f.write(f"path: {DST}/{name}\ntrain: images/train\nval: images/val\nnc: 1\nchannels: 3\nnames:\n  - human\n")
    for k, v in counts.items():
        print(f"{k:32s} {v}")
    ok = all(counts[f"{p}/images{s}/train"] == 6957 and counts[f"{p}/images{s}/val"] == 1740
             for p, s in (("fusion", "/rgb"), ("baseline_rgb", ""), ("baseline_ir", "")))
    print("image counts 6957/1740 everywhere:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
