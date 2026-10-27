import os
import glob

# Filter out targets with area < 0.02 (2% of image, e.g. humans far away)
# Typical COCO medium/large threshold mapped to normalized coordinates
AREA_THRESH = 0.02

labels_dir = "/Users/test/Q-WildFusion/phase3/dataset_aligned/labels/rgb/val"
label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

filtered_count = 0
total_count = 0

for label_file in label_files:
    valid_lines = []
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            total_count += 1
            w = float(parts[3])
            h = float(parts[4])
            if (w * h) >= AREA_THRESH:
                valid_lines.append(line)
            else:
                filtered_count += 1
                
    with open(label_file, "w") as f:
        f.writelines(valid_lines)

print(f"Filtered out {filtered_count} small objects from {total_count} total objects.")
print(f"Remaining for Medium/Large evaluation: {total_count - filtered_count}")
