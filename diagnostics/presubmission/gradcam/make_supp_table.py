#!/usr/bin/env python3
"""
Supplementary table: CAM energy inside GT boxes relative to chance, per level (P3/P4/P5) and Sum.

Computed from energy_per_image.csv (no CAM recomputation). Every map, per level and Sum, was measured on the
768x960 upsampled grid with the full-resolution GT box mask. Per (method, model, layer group) all four columns use
Sum's image set and Sum's chance level; a per-level map that is all zero (ReLU leaves nothing; energy fraction 0/0)
counts as fraction 0, because it puts no energy inside the boxes.

  .venv/bin/python diagnostics/presubmission/gradcam/make_supp_table.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper")
MODELS = {"complete": "Full model", "cbam_add": "Fusion baseline (CBAM + add)",
          "baseline_thermal": "Thermal-only YOLO11n", "baseline_rgb": "RGB-only YOLO11n"}
METHODS = {"gradcam": "Grad-CAM", "eigencam": "EigenCAM"}
LAYERS = ["RGB features", "Thermal features", "Fused", "Neck output"]  # row order within a model
LEVELS = ("p3", "p4", "p5", "sum")


def layer_name(model, group):
    if group == "bb":  # single-modality baselines: backbone layers 4/6/10 of their own modality
        return "Thermal features" if model == "baseline_thermal" else "RGB features"
    return {"rgb": "RGB features", "ir": "Thermal features", "fused": "Fused", "neck": "Neck output"}[group]


def build():
    d = pd.read_csv(os.path.join(HERE, "energy_per_image.csv"))
    d["group"] = d.layer.str.rsplit("_", n=1).str[0]
    d["level"] = d.layer.str.rsplit("_", n=1).str[1]
    rows = []
    for (meth, model, grp), g in d.groupby(["method", "model", "group"]):
        w = g.pivot(index="image", columns="level", values="energy_in_gt")
        area = g.groupby("image").gt_area_frac.first()
        base = w["sum"].notna()  # Sum's image set
        chance = area[base].mean()
        r = {"Method": METHODS[meth], "Model": MODELS[model], "Layer": layer_name(model, grp)}
        r.update({l.upper() if l != "sum" else "Sum": w.loc[base, l].fillna(0.0).mean() / chance for l in LEVELS})
        r.update({"n images": int(base.sum()), "Chance": chance,
                  "empty P3/P4/P5": "/".join(str(int((base & w[l].isna()).sum())) for l in ("p3", "p4", "p5"))})
        rows.append(r)
    t = pd.DataFrame(rows)
    order = {"Method": list(METHODS.values()), "Model": list(MODELS.values()), "Layer": LAYERS}
    t = t.sort_values(by=list(order), key=lambda c: c.map({v: i for i, v in enumerate(order[c.name])}))
    return t.reset_index(drop=True)


CAPTION = ("Supplementary Table. CAM energy inside GT boxes relative to chance. Ratio = mean fraction of CAM energy "
           "inside GT boxes / mean fraction of image area covered by GT boxes (Chance); 1.0 = chance. All ratios are "
           "computed on maps upsampled to 768x960 with the same full-resolution GT box mask; Sum = the P3+P4+P5 maps "
           "summed after upsampling, so each image's Sum fraction is the energy-weighted average of its three per-level "
           "fractions. Within a row, P3/P4/P5 and Sum use the same images and the same Chance. Val set, 1740 images; "
           "Grad-CAM excludes images with no target prediction (conf>=0.25, IoU>=0.5 to a GT box), hence n < 1740 and "
           "Chance 0.0340 for the RGB-only model. A per-level Grad-CAM map that is entirely zero after ReLU counts as "
           "fraction 0 (counts in the CSV column 'empty P3/P4/P5'). Single-modality baselines: 'RGB features' / "
           "'Thermal features' = their backbone (layers 4/6/10) in the same position as the fusion models' branch "
           "features; they have no Fused row.")


def write(t):
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "supp_table_energy_ratios.csv"), index=False, float_format="%.4f")
    md = [CAPTION, "", "| Method | Model | Layer | P3 | P4 | P5 | Sum | n images | Chance |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r.Method} | {r.Model} | {r.Layer} | {r.P3:.2f} | {r.P4:.2f} | {r.P5:.2f} | {r.Sum:.2f} | "
                  f"{r['n images']} | {r.Chance:.4f} |")
    open(os.path.join(OUT, "supp_table_energy_ratios.md"), "w").write("\n".join(md) + "\n")
    tex = ["% " + CAPTION, r"\begin{tabular}{lllrrrrrr}", r"\toprule",
           r"Method & Model & Layer & P3 & P4 & P5 & Sum & $n$ & Chance \\", r"\midrule"]
    prev = None
    for _, r in t.iterrows():
        if prev is not None and r.Method != prev:
            tex.append(r"\midrule")
        prev = r.Method
        tex.append(f"{r.Method} & {r.Model} & {r.Layer} & {r.P3:.2f} & {r.P4:.2f} & {r.P5:.2f} & {r.Sum:.2f} & "
                   f"{r['n images']} & {r.Chance:.4f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "supp_table_energy_ratios.tex"), "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    t = build()
    write(t)
    print(t.round(3).to_string(index=False))
