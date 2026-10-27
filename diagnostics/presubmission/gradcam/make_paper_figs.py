#!/usr/bin/env python3
"""
Paper figures (Grad-CAM only) for 150202 and 250097, plus the supplementary energy table.

CAM panels reuse the saved maps (figures/<stem>_maps_q.npz: float16, 1/4 res 192x240), upsampled bilinearly
to 768x960 for display. Nothing about the CAMs is recomputed. The detections panel needs the post-NMS boxes,
which were not saved, so they come from one no-grad forward pass + NMS per model (same checkpoint and input
as the figures run); the box counts are checked against figures/figures_meta.json.

  .venv/bin/python diagnostics/presubmission/gradcam/make_paper_figs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import gradcam_paired as G  # noqa: E402

IMAGES = ["150202.jpg", "250097.jpg"]
ROWS = {"complete": "Full model", "cbam_add": "Fusion baseline\n(CBAM + add)",
        "baseline_thermal": "Thermal-only\nYOLO11n", "baseline_rgb": "RGB-only\nYOLO11n"}
OUT = os.path.join(HERE, "paper")
FS = 7  # pt, minimum font size everywhere
CMAP = "inferno"
plt.rcParams.update({"font.size": FS, "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.4})


def load_maps(stem):
    z = np.load(os.path.join(HERE, "figures", f"{stem}_maps_q.npz"))

    def get(model, layer):
        k = f"{model}__gradcam__{layer}"
        if k not in z.files:
            return None
        m = torch.from_numpy(z[k].astype(np.float32))[None, None]
        return F.interpolate(m, size=(G.H, G.W), mode="bilinear", align_corners=False)[0, 0].numpy()
    return get


def detections(name, models, datas, device):
    out = {}
    with torch.no_grad():
        for key, m in models.items():
            d = datas[key]
            if d.fusion:
                rgb, ir, gt = d.get(name)
                y = m(G.RGBTBatch((rgb.to(device), ir.to(device))))
            else:
                x, _, gt = d.get(name)
                y = m(x.to(device))
            y = y[0] if isinstance(y, (list, tuple)) else y
            out[key] = (G.non_max_suppression(y.float(), conf_thres=G.CONF, iou_thres=G.NMS_IOU)[0].cpu(), gt)
    return out


def sci(v, _=None):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.1f}e{e}"


COLS = ["RGB input", "IR input", "RGB features", "Thermal features", None, None, "Neck output", None, None,
        "Detections"]
WR = [1, 1, 1, 1, 0.05, 0.42, 1, 0.05, 0.42, 1]
WIDTH_IN = 17.5 / 2.54
LEFT_IN, RIGHT_IN, HEAD_IN, LEGEND_IN, GAP_IN = 0.62, 0.04, 0.2, 0.32, 0.05
UNIT = (WIDTH_IN - LEFT_IN - RIGHT_IN) / (sum(WR) * 1.0 + 0.02 * (len(WR) - 1))
PH = UNIT * G.H / G.W
BLOCK_IN = len(ROWS) * PH + (len(ROWS) - 1) * GAP_IN  # image rows of one block, without its header


def draw_block(fig, height_in, top_in, maps, dets, rgb_img, ir_img):
    """One 4-row block; its column headers occupy HEAD_IN above top_in (inches from the figure top)."""
    cols, wr = COLS, WR
    gs = fig.add_gridspec(len(ROWS), len(cols), width_ratios=wr, wspace=0.02, hspace=GAP_IN / PH,
                          left=LEFT_IN / WIDTH_IN, right=1 - RIGHT_IN / WIDTH_IN,
                          top=1 - top_in / height_in, bottom=1 - (top_in + BLOCK_IN) / height_in)
    ranges = {}
    for r, key in enumerate(ROWS):
        fusion = not key.startswith("baseline")
        if fusion:
            feat = [maps(key, "rgb_sum"), maps(key, "ir_sum")]
        elif key == "baseline_rgb":
            feat = [maps(key, "bb_sum"), None]
        else:
            feat = [None, maps(key, "bb_sum")]
        neck = maps(key, "neck_sum")
        va = max(float(f.max()) for f in feat if f is not None)
        vb = float(neck.max())
        ranges[key] = {"features": [0.0, va], "neck": [0.0, vb]}
        used_rgb, used_ir = fusion or key == "baseline_rgb", fusion or key == "baseline_thermal"
        ax = [fig.add_subplot(gs[r, c]) for c in range(len(cols))]
        ax[0].imshow(rgb_img if used_rgb else rgb_img * 0.25)
        ax[1].imshow(ir_img if used_ir else ir_img * 0.25, cmap="gray", vmin=0, vmax=1)
        for c, used in ((0, used_rgb), (1, used_ir)):
            if not used:
                ax[c].text(0.5, 0.5, "not an input", ha="center", va="center", transform=ax[c].transAxes,
                           color="white", fontsize=FS)
        for c, p, bg, vmax in ((2, feat[0], rgb_img, va), (3, feat[1], ir_img, va), (6, neck, ir_img, vb)):
            if p is None:
                ax[c].set_facecolor("#dddddd")
                ax[c].text(0.5, 0.5, "not an\ninput", ha="center", va="center", transform=ax[c].transAxes,
                           color="#555555", fontsize=FS)
            else:
                ax[c].imshow(bg, cmap="gray" if bg.ndim == 2 else None)
                ax[c].imshow(p, cmap=CMAP, alpha=0.55, vmin=0, vmax=vmax, interpolation="bilinear")
        for c, vmax in ((4, va), (7, vb)):
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax)), cax=ax[c],
                              ticks=[0, vmax], format=FuncFormatter(sci))
            cb.ax.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
            cb.outline.set_linewidth(0.4)
            lo, hi = cb.ax.get_yticklabels()  # keep both labels inside the row (no clash with neighbours)
            lo.set_va("bottom")
            hi.set_va("top")
        for c in (5, 8):
            ax[c].axis("off")
        d, gt = dets[key]
        a = ax[9]
        a.imshow(ir_img, cmap="gray", vmin=0, vmax=1)
        for x1, y1, x2, y2 in gt.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=0.7))
        for x1, y1, x2, y2, _, _ in d.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=0.7, ls="--"))
        ax[0].set_ylabel(ROWS[key], fontsize=FS, labelpad=3)
        for c, t in enumerate(cols):
            if c in (4, 5, 7, 8):
                continue
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            for s in ax[c].spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax[c].set_title(t, fontsize=FS, pad=2)
    return ranges


def add_legend(fig):
    handles = [Line2D([], [], color="red", lw=0.9, ls="--", label="Prediction (conf \u2265 0.25)"),
               Line2D([], [], color="lime", lw=0.9, label="Ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=FS,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.5)


def paper_figure(name, maps, dets, rgb_img, ir_img, meta):
    stem = os.path.splitext(name)[0]
    width_in = WIDTH_IN
    height_in = HEAD_IN + BLOCK_IN + LEGEND_IN
    fig = plt.figure(figsize=(width_in, height_in))
    ranges = draw_block(fig, height_in, HEAD_IN, maps, dets, rgb_img, ir_img)
    add_legend(fig)
    png, pdf = os.path.join(OUT, f"fig_gradcam_{stem}.png"), os.path.join(OUT, f"fig_gradcam_{stem}.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    lines = [f"{name} -- Grad-CAM paper figure ({width_in * 2.54:.1f} x {height_in * 2.54:.1f} cm).",
             "Rows: Full model (complete), Fusion baseline (CBAM + add), Thermal-only YOLO11n, RGB-only YOLO11n; "
             "seed 0, last.pt (EMA, epoch 12).",
             "Columns: RGB input | IR input | RGB features | Thermal features | Neck output | Detections. "
             "Features = post-attention backbone features entering fusion (single-modality baselines: backbone "
             "layers 4/6/10 in their own modality's column); Neck output = Detect inputs; each the sum of the "
             "P3/P4/P5 Grad-CAMs, from the saved 1/4-resolution maps upsampled to 768x960.",
             "Grad-CAM target: sum of person scores of raw pre-NMS predictions with conf>=0.25 and IoU>=0.5 to a GT box.",
             "Colour scales per row: one for RGB + Thermal features, one for Neck output (fused column dropped, so "
             "the feature scale is the max over the two feature panels shown). Ranges [min, max]:"]
    for k, rg in ranges.items():
        lines.append(f"  {ROWS[k].replace(chr(10), ' '):34s} features [0, {rg['features'][1]:.4g}]   "
                     f"neck [0, {rg['neck'][1]:.4g}]   target preds={meta[k]['n_target']}  dets={len(dets[k][0])}  "
                     f"GT={len(dets[k][1])}")
    open(os.path.join(OUT, f"fig_gradcam_{stem}_caption.txt"), "w").write("\n".join(lines) + "\n")
    return png, pdf, (width_in * 2.54, height_in * 2.54)


def combined_figure(blocks):
    """(a) on top, (b) below; one legend at the bottom. blocks = [(label, maps, dets, rgb_img, ir_img), ...]"""
    label_in = 0.06  # small extra space above each block for the panel letter
    height_in = len(blocks) * (label_in + HEAD_IN + BLOCK_IN) + (len(blocks) - 1) * 0.08 + LEGEND_IN
    fig = plt.figure(figsize=(WIDTH_IN, height_in))
    y = 0.0
    for i, (label, maps, dets, rgb_img, ir_img) in enumerate(blocks):
        if i:
            y += 0.08
        fig.text(0.004, 1 - y / height_in, label, fontsize=8, fontweight="bold", ha="left", va="top")
        draw_block(fig, height_in, y + label_in + HEAD_IN, maps, dets, rgb_img, ir_img)
        y += label_in + HEAD_IN + BLOCK_IN
    add_legend(fig)
    png, pdf = os.path.join(OUT, "fig_gradcam_paper.png"), os.path.join(OUT, "fig_gradcam_paper.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    return png, (WIDTH_IN * 2.54, height_in * 2.54)


def energy_table():
    """Superseded by make_supp_table.py (per-level ratios on Sum's image set and chance)."""
    import make_supp_table
    t = make_supp_table.build()
    make_supp_table.write(t)
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    device = G.pick_device("auto")
    meta = json.load(open(os.path.join(HERE, "figures", "figures_meta.json")))
    datas = {k: G.ValData(k) for k in ROWS}
    models = {k: G.load_model(k, device) for k in ROWS}
    blocks = []
    for name, label in zip(IMAGES, ("(a)", "(b)")):
        dets = detections(name, models, datas, device)
        for k in ROWS:  # same boxes as the figures run
            assert len(dets[k][0]) == meta[name][k]["n_dets"], (name, k, len(dets[k][0]), meta[name][k]["n_dets"])
            assert len(dets[k][1]) == meta[name][k]["n_gt"]
        rgb, ir, _ = datas["complete"].get(name)
        png, pdf, size = paper_figure(name, load_maps(os.path.splitext(name)[0]), dets,
                                      rgb[0].permute(1, 2, 0).numpy(), ir[0, 0].numpy(), meta[name])
        print(f"[paper] {name}: {png} + .pdf  {size[0]:.1f} x {size[1]:.1f} cm")
        blocks.append((label, load_maps(os.path.splitext(name)[0]), dets, rgb[0].permute(1, 2, 0).numpy(),
                       ir[0, 0].numpy()))
    png, size = combined_figure(blocks)
    print(f"[paper] combined: {png} + .pdf  {size[0]:.1f} x {size[1]:.1f} cm")
    t = energy_table()
    print(f"[paper] supplementary table: {len(t)} rows")


if __name__ == "__main__":
    main()
