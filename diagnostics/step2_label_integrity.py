"""
Step 2: label/data integrity check for the val split actually used by the
baseline run (phase3/dataset_aligned/images/rgb/val + labels/rgb/val).
Read-only; no files modified.
"""
import os

IMG_DIR = "phase3/dataset_aligned/images/rgb/val"
LBL_DIR = "phase3/dataset_aligned/labels/rgb/val"

images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
print(f"val images found: {len(images)}")

missing_labels = []
bad_coords = []
bad_class = []
total_boxes = 0
images_with_boxes = 0
empty_label_files = 0

for img_name in images:
    stem = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(LBL_DIR, stem + ".txt")
    if not os.path.isfile(lbl_path):
        missing_labels.append(img_name)
        continue

    with open(lbl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        empty_label_files += 1
        continue

    images_with_boxes += 1
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            bad_coords.append((lbl_path, line, "wrong field count"))
            continue
        cls, x, y, w, h = parts
        total_boxes += 1
        if cls != "0":
            bad_class.append((lbl_path, cls))
        for name, v in [("x", x), ("y", y), ("w", w), ("h", h)]:
            fv = float(v)
            if not (0.0 <= fv <= 1.0):
                bad_coords.append((lbl_path, line, f"{name}={fv} out of [0,1]"))

print(f"images missing a label file: {len(missing_labels)}")
if missing_labels[:10]:
    print("  sample:", missing_labels[:10])
print(f"empty label files (0 boxes): {empty_label_files}")
print(f"images with >=1 box: {images_with_boxes}")
print(f"total GT boxes: {total_boxes}")
print(f"label lines with out-of-[0,1] coords: {len(bad_coords)}")
if bad_coords[:10]:
    print("  sample:", bad_coords[:10])
print(f"label lines with non-zero class index: {len(bad_class)}")
if bad_class[:10]:
    print("  sample:", bad_class[:10])

# Also check that every label file corresponds to an actual image (reverse direction)
label_files = sorted(f for f in os.listdir(LBL_DIR) if f.endswith(".txt") and f != "val.cache")
image_stems = {os.path.splitext(f)[0] for f in images}
orphan_labels = [f for f in label_files if os.path.splitext(f)[0] not in image_stems]
print(f"label files with no matching image: {len(orphan_labels)}")
if orphan_labels[:10]:
    print("  sample:", orphan_labels[:10])

print()
print("VERDICT: label/data integrity for val split —",
      "CLEAN" if not missing_labels and not bad_coords and not bad_class and not orphan_labels else "ISSUES FOUND (see above)")
