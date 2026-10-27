#!/usr/bin/env python3
"""
Clean Grad-CAM examples of correct detections (neck P4 only), 6 images x 4 models.

Selection (recorded in paper/correct_examples/selection_correct.json):
  1. all 1740 val images; keep those with 1-3 GT persons where EVERY model detects EVERY GT person with no false
     positive. Detections: NMS conf>=0.25, IoU 0.7. Matching: greedy one-to-one by IoU (highest first), a match needs
     IoU>=0.5; all GT matched and all detections matched <=> correct.
  2. rank by the full model's Grad-CAM neck P4 in-box energy fraction from energy_per_image.csv (as in the
     regenerated table: an all-zero map counts as 0), descending; ties by image id.
  3. top 6.
Maps: Grad-CAM at neck P4 (Detect input, stride 16), same target as gradcam_paired.py (person scores of raw pre-NMS
predictions with conf>=0.25 and IoU>=0.5 to a GT box), computed fresh through the paired path with run_model().

  .venv/bin/python diagnostics/presubmission/gradcam/make_correct_examples.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import gradcam_paired as G  # noqa: E402

OUT = os.path.join(HERE, "paper", "correct_examples")
COLS = {"complete": "Full model", "cbam_add": "Fusion baseline\n(CBAM + add)",
        "baseline_thermal": "Thermal-only\nYOLO11n", "baseline_rgb": "RGB-only\nYOLO11n"}
FS, N_PICK, ALPHA = 7, 6, 0.5
DET_CACHE = os.path.join(OUT, "detections_val.json")
plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.4})


def match(gt, det):
    """Greedy one-to-one matching at IoU>=0.5. Returns (#GT matched, #det matched)."""
    if not len(gt) or not len(det):
        return 0, 0
    iou = G.box_iou(gt, det[:, :4])
    pairs = sorted(((float(iou[i, j]), i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])
                    if iou[i, j] >= G.IOU_MATCH), reverse=True)
    gi, dj = set(), set()
    for _, i, j in pairs:
        if i not in gi and j not in dj:
            gi.add(i)
            dj.add(j)
    return len(gi), len(dj)


def forward_x(key, d, name, device):
    if d.fusion:
        rgb, ir, gt = d.get(name)
        return G.RGBTBatch((rgb.to(device), ir.to(device))), gt
    x, _, gt = d.get(name)
    return x.to(device), gt


def detect_all(datas, models, device):
    if os.path.isfile(DET_CACHE):
        return json.load(open(DET_CACHE))
    names = datas["complete"].names()
    res, t0 = {}, time.time()
    with torch.no_grad():
        for i, n in enumerate(names):
            res[n] = {}
            for key, m in models.items():
                x, gt = forward_x(key, datas[key], n, device)
                y = m(x)
                y = y[0] if isinstance(y, (list, tuple)) else y
                det = G.non_max_suppression(y.float(), conf_thres=G.CONF, iou_thres=G.NMS_IOU)[0].cpu()
                gm, dm = match(gt, det)
                res[n][key] = {"n_gt": len(gt), "n_det": len(det), "gt_matched": gm, "det_matched": dm,
                               "det": [[round(v, 2) for v in r] for r in det[:, :5].tolist()],
                               "gt": [[round(v, 2) for v in r] for r in gt.tolist()]}
            if i % 200 == 0:
                print(f"[detect] {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(DET_CACHE, "w"))
    return res


def select(dets):
    passed = []
    for n, r in dets.items():
        n_gt = r["complete"]["n_gt"]
        if not 1 <= n_gt <= 3:
            continue
        if all(v["n_gt"] == n_gt and v["gt_matched"] == n_gt and v["det_matched"] == v["n_det"] == n_gt
               for v in r.values()):
            passed.append(n)
    e = pd.read_csv(os.path.join(HERE, "energy_per_image.csv"))
    e = e[(e.method == "gradcam") & (e.model == "complete") & (e.layer == "neck_p4")].set_index("image")
    score = {n: (0.0 if pd.isna(e.loc[n, "energy_in_gt"]) else float(e.loc[n, "energy_in_gt"])) for n in passed}
    ranked = sorted(passed, key=lambda n: (-score[n], n))
    sel = {
        "rule": ("All 1740 val images. Keep images with 1-3 GT persons where each of the 4 models (complete, cbam_add, "
                 "baseline_thermal, baseline_rgb; seed 0 last.pt) detects every GT person with no false positive: "
                 "detections = NMS conf>=0.25, IoU 0.7; greedy one-to-one matching by descending IoU, match iff "
                 "IoU>=0.5; pass iff all GT and all detections are matched. Rank passing images by the full model's "
                 "Grad-CAM neck P4 in-box energy fraction from energy_per_image.csv (all-zero map counts as 0, as in "
                 "the regenerated supplementary table), descending, ties by image id. Take the top 6."),
        "n_val": len(dets), "n_passed": len(passed),
        "n_with_1_to_3_gt": sum(1 <= r["complete"]["n_gt"] <= 3 for r in dets.values()),
        "selected": [{"image": n, "rank": i + 1, "complete_neck_p4_energy_in_gt": score[n],
                      "n_gt": dets[n]["complete"]["n_gt"]} for i, n in enumerate(ranked[:N_PICK])],
        "ranked_all_passed": [[n, score[n]] for n in ranked],
    }
    json.dump(sel, open(os.path.join(OUT, "selection_correct.json"), "w"), indent=1)
    return sel


def neck_p4(names, datas, models, device):
    out = {}
    for n in names:
        out[n] = {}
        for key, m in models.items():
            d = datas[key]
            if d.fusion:
                rgb, ir, gt = d.get(n)
            else:
                x, _, gt = d.get(n)
                rgb, ir = (x, None) if key == "baseline_rgb" else (None, x)
            acts, dets, n_t, _ = G.run_model(m, key, rgb, ir, gt, device)
            a, g = acts["neck_p4"]
            assert n_t > 0 and g is not None, (n, key)
            cam = G.gradcam(a, g)
            mask = G.box_mask(gt)
            out[n][key] = {"cam": cam.numpy(), "dets": dets, "gt": gt,
                           "energy_in_gt": float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else 0.0}
    return out


def draw_panel(ax, ir_img, cam, dets, gt):
    ax.imshow(ir_img, cmap="gray", vmin=0, vmax=1)
    vmax = float(cam.max())
    ax.imshow(cam / vmax if vmax > 0 else cam, cmap="inferno", alpha=ALPHA, vmin=0, vmax=1, interpolation="bilinear")
    for x1, y1, x2, y2 in gt.tolist():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=0.8))
    for x1, y1, x2, y2, _, _ in dets.tolist():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=0.8, ls="--"))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def figure(names, maps, irs):
    width_in = 17.5 / 2.54
    wr = [1, 1, 1, 1, 0.05]
    left, right, top, bottom, ws = 0.04, 0.42, 0.3, 0.04, 0.03
    unit = (width_in - left - right) / (sum(wr) + ws * (len(wr) - 1))
    ph = unit * G.H / G.W
    gap = 0.04
    height_in = top + bottom + len(names) * ph + (len(names) - 1) * gap
    fig = plt.figure(figsize=(width_in, height_in))
    gs = fig.add_gridspec(len(names), len(wr), width_ratios=wr, wspace=ws, hspace=gap / ph,
                          left=left / width_in, right=1 - right / width_in, top=1 - top / height_in,
                          bottom=bottom / height_in)
    for r, n in enumerate(names):
        for c, key in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            mm = maps[n][key]
            draw_panel(ax, irs[n], mm["cam"], mm["dets"], mm["gt"])
            if r == 0:
                ax.set_title(COLS[key], fontsize=FS, pad=3)
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1)), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["0", "max"], fontsize=FS)
    cb.set_label("relative Grad-CAM", fontsize=FS, labelpad=2)
    cb.ax.tick_params(length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    png, pdf = os.path.join(OUT, "fig_gradcam_correct.png"), os.path.join(OUT, "fig_gradcam_correct.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return png, (width_in * 2.54, height_in * 2.54)


def panels(names, maps, irs):
    d = os.path.join(OUT, "panels")
    os.makedirs(d, exist_ok=True)
    for n in names:
        for key in COLS:
            fig = plt.figure(figsize=(G.W / 300, G.H / 300))  # native 960x768 px at 300 dpi
            ax = fig.add_axes([0, 0, 1, 1])
            mm = maps[n][key]
            draw_panel(ax, irs[n], mm["cam"], mm["dets"], mm["gt"])
            for s in ax.spines.values():
                s.set_visible(False)
            fig.savefig(os.path.join(d, f"{os.path.splitext(n)[0]}_{key}.png"), dpi=300)
            plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = G.pick_device("auto")
    datas = {k: G.ValData(k) for k in COLS}
    models = {k: G.load_model(k, device) for k in COLS}
    dets = detect_all(datas, models, device)
    sel = select(dets)
    names = [s["image"] for s in sel["selected"]]
    print(f"[select] {sel['n_with_1_to_3_gt']} images with 1-3 GT; {sel['n_passed']} passed; top {N_PICK}: {names}")
    maps = neck_p4(names, datas, models, device)
    # consistency: fresh full-model P4 fraction must equal the energy_per_image.csv value used for ranking
    for s in sel["selected"]:
        fresh = maps[s["image"]]["complete"]["energy_in_gt"]
        assert abs(fresh - s["complete_neck_p4_energy_in_gt"]) < 1e-4, (s["image"], fresh, s)
        s["fresh_recomputed"] = fresh
    json.dump(sel, open(os.path.join(OUT, "selection_correct.json"), "w"), indent=1)
    irs = {n: datas["complete"].get(n)[1][0, 0].numpy() for n in names}
    for n in names:  # the detections drawn are the ones the filter used
        for key in COLS:
            assert len(maps[n][key]["dets"]) == dets[n][key]["n_det"], (n, key)
    np.savez_compressed(os.path.join(OUT, "neck_p4_gradcam.npz"),
                        **{f"{os.path.splitext(n)[0]}__{k}": maps[n][k]["cam"].astype(np.float16) for n in names
                           for k in COLS})
    png, size = figure(names, maps, irs)
    panels(names, maps, irs)
    lines = ["Grad-CAM at the neck P4 layer (Detect input, stride 16) for correct detections. Rows (top to bottom): " +
             ", ".join(f"{os.path.splitext(n)[0]}" for n in names) + ".",
             "Columns: Full model, Fusion baseline (CBAM + add), Thermal-only YOLO11n, RGB-only YOLO11n (seed 0, "
             "last.pt = EMA, epoch 12). Overlay on the thermal image for every model, including RGB-only.",
             "Each panel normalised to its own max (colourbar = relative Grad-CAM, 0 to max). Predictions: NMS "
             "conf>=0.25, IoU 0.7 (dashed red); GT green. Target: person scores of raw pre-NMS predictions with "
             "conf>=0.25 and IoU>=0.5 to a GT box.",
             f"Selection: {sel['n_passed']} of 1740 val images pass (1-3 GT persons, all four models detect every GT "
             "with no false positive); top 6 by the full model's neck-P4 in-box energy fraction. See "
             "selection_correct.json.", "", "Per-panel neck P4 in-box energy fraction:"]
    for n in names:
        lines.append(f"  {os.path.splitext(n)[0]}: " + "  ".join(f"{k}={maps[n][k]['energy_in_gt']:.3f}" for k in COLS))
    open(os.path.join(OUT, "fig_gradcam_correct_caption.txt"), "w").write("\n".join(lines) + "\n")
    print(f"[fig] {png} + .pdf  {size[0]:.1f} x {size[1]:.1f} cm")


if __name__ == "__main__":
    main()
