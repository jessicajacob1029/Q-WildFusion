#!/usr/bin/env python3
"""
Supplementary versions of gradcam_all.pdf / eigencam_all.pdf with the paper labels.

CAM panels: saved maps (figures/<stem>_maps_q.npz, float16, 1/4 res) upsampled to 768x960; no CAM is recomputed.
Detections: post-NMS boxes were never saved, so they come from one no-grad forward pass per model (same checkpoint
and input as the figures run); counts are checked against figures/figures_meta.json.

  .venv/bin/python diagnostics/presubmission/gradcam/make_supp_figs.py
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import gradcam_paired as G  # noqa: E402
from make_paper_figs import FS, ROWS, detections, sci  # noqa: E402  (same labels, font size, tick format)

OUT = os.path.join(HERE, "paper")
CMAP = "inferno"
COLS = ["RGB input", "IR input", "RGB\nfeatures", "Thermal\nfeatures", "Fused\nfeatures", None, None, "Neck\noutput",
        None, None, "Detections"]
# same wording as the paper rows, wrapped to fit horizontally in the left margin
ROW_LABELS = {"complete": "Full\nmodel", "cbam_add": "Fusion\nbaseline\n(CBAM + add)",
              "baseline_thermal": "Thermal-only\nYOLO11n", "baseline_rgb": "RGB-only\nYOLO11n"}
WR = [1, 1, 1, 1, 1, 0.05, 0.6, 1, 0.05, 0.6, 1]
WIDTH_IN = 17.5 / 2.54
LEFT_IN, RIGHT_IN, LABEL_IN, HEAD_IN, LEGEND_IN, GAP_IN = 0.8, 0.04, 0.16, 0.3, 0.32, 0.05
UNIT = (WIDTH_IN - LEFT_IN - RIGHT_IN) / (sum(WR) + 0.02 * (len(WR) - 1))
PH = UNIT * G.H / G.W
HEIGHT_IN = LABEL_IN + HEAD_IN + len(ROWS) * PH + (len(ROWS) - 1) * GAP_IN + LEGEND_IN
METHOD = {"gradcam": "Grad-CAM", "eigencam": "EigenCAM"}


def load_maps(stem, meth):
    z = np.load(os.path.join(HERE, "figures", f"{stem}_maps_q.npz"))

    def get(model, layer):
        k = f"{model}__{meth}__{layer}"
        if k not in z.files:
            return None
        m = torch.from_numpy(z[k].astype(np.float32))[None, None]
        return F.interpolate(m, size=(G.H, G.W), mode="bilinear", align_corners=False)[0, 0].numpy()
    return get


def grey(ax, text):
    ax.set_facecolor("#dddddd")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#555555", fontsize=FS,
            wrap=True)


def page(name, tag, meth, maps, dets, rgb_img, ir_img, meta):
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    top_in = LABEL_IN + HEAD_IN
    gs = fig.add_gridspec(len(ROWS), len(COLS), width_ratios=WR, wspace=0.02, hspace=GAP_IN / PH,
                          left=LEFT_IN / WIDTH_IN, right=1 - RIGHT_IN / WIDTH_IN, top=1 - top_in / HEIGHT_IN,
                          bottom=LEGEND_IN / HEIGHT_IN)
    fig.text(0.004, 1 - 0.02 / HEIGHT_IN, f"{os.path.splitext(name)[0]} ({tag})", fontsize=FS, ha="left", va="top")
    ranges = {}
    for r, key in enumerate(ROWS):
        fusion = not key.startswith("baseline")
        used_rgb, used_ir = fusion or key == "baseline_rgb", fusion or key == "baseline_thermal"
        no_target = meth == "gradcam" and meta[key]["n_target"] == 0
        if fusion:
            feat = [maps(key, "rgb_sum"), maps(key, "ir_sum"), maps(key, "fused_sum")]
        elif key == "baseline_rgb":
            feat = [maps(key, "bb_sum"), None, None]
        else:
            feat = [None, maps(key, "bb_sum"), None]
        neck = maps(key, "neck_sum")
        if no_target:  # Grad-CAM undefined: nothing was saved for this row
            assert all(f is None for f in feat) and neck is None, (name, key)
        have_feat = [f for f in feat if f is not None]
        va = max(float(f.max()) for f in have_feat) if have_feat else None
        vb = float(neck.max()) if neck is not None else None
        ranges[key] = {"features": va, "neck": vb, "n_target": meta[key]["n_target"]}
        ax = [fig.add_subplot(gs[r, c]) for c in range(len(COLS))]
        ax[0].imshow(rgb_img if used_rgb else rgb_img * 0.25)
        ax[1].imshow(ir_img if used_ir else ir_img * 0.25, cmap="gray", vmin=0, vmax=1)
        for c, used in ((0, used_rgb), (1, used_ir)):
            if not used:
                ax[c].text(0.5, 0.5, "Not an input", ha="center", va="center", transform=ax[c].transAxes,
                           color="white", fontsize=FS)
        panels = ((2, feat[0], rgb_img, va, "Not an input" if not used_rgb else None),
                  (3, feat[1], ir_img, va, "Not an input" if not used_ir else None),
                  (4, feat[2], ir_img, va, "No fusion" if not fusion else None),
                  (7, neck, ir_img, vb, None))
        for c, p, bg, vmax, why in panels:
            if why is not None:
                grey(ax[c], why)
            elif p is None:
                grey(ax[c], "No qualifying\nprediction")
            else:
                ax[c].imshow(bg, cmap="gray" if bg.ndim == 2 else None)
                ax[c].imshow(p, cmap=CMAP, alpha=0.55, vmin=0, vmax=vmax, interpolation="bilinear")
        for c, vmax in ((5, va), (8, vb)):
            if vmax is None:
                ax[c].axis("off")
                continue
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax)), cax=ax[c],
                              ticks=[0, vmax], format=FuncFormatter(sci))
            cb.ax.tick_params(labelsize=FS, length=1.5, pad=1, width=0.4)
            cb.outline.set_linewidth(0.4)
            lo, hi = cb.ax.get_yticklabels()
            lo.set_va("bottom")
            hi.set_va("top")
        for c in (6, 9):
            ax[c].axis("off")
        d, gt = dets[key]
        a = ax[10]
        a.imshow(ir_img, cmap="gray", vmin=0, vmax=1)
        for x1, y1, x2, y2 in gt.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=0.7))
        for x1, y1, x2, y2, _, _ in d.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=0.7, ls="--"))
        ax[0].set_ylabel(ROW_LABELS[key], fontsize=FS, rotation=0, ha="right", va="center", labelpad=4)
        for c, t in enumerate(COLS):
            if t is None:
                continue
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            for s in ax[c].spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax[c].set_title(t, fontsize=FS, pad=2)
    handles = [Line2D([], [], color="red", lw=0.9, ls="--", label="Prediction (conf ≥ 0.25)"),
               Line2D([], [], color="lime", lw=0.9, label="Ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=FS, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.5)
    return fig, ranges


def main():
    device = G.pick_device("auto")
    sel = json.load(open(os.path.join(HERE, "selection.json")))
    meta = json.load(open(os.path.join(HERE, "figures", "figures_meta.json")))
    miss = {m["image"] for m in sel["miss"]}
    datas = {k: G.ValData(k) for k in ROWS}
    models = {k: G.load_model(k, device) for k in ROWS}
    pdfs = {m: PdfPages(os.path.join(OUT, f"supp_fig_{m}.pdf")) for m in METHOD}
    cap = {m: [] for m in METHOD}
    for name in sel["images"]:
        tag = "miss case" if name in miss else "random"
        dets = detections(name, models, datas, device)
        for k in ROWS:
            assert len(dets[k][0]) == meta[name][k]["n_dets"], (name, k, len(dets[k][0]), meta[name][k]["n_dets"])
            assert len(dets[k][1]) == meta[name][k]["n_gt"]
        rgb, ir, _ = datas["complete"].get(name)
        rgb_img, ir_img = rgb[0].permute(1, 2, 0).numpy(), ir[0, 0].numpy()
        stem = os.path.splitext(name)[0]
        for meth in METHOD:
            fig, ranges = page(name, tag, meth, load_maps(stem, meth), dets, rgb_img, ir_img, meta[name])
            pdfs[meth].savefig(fig, dpi=300)
            plt.close(fig)
            cap[meth].append(f"{stem} ({tag})")
            for k, rg in ranges.items():
                f = "n/a" if rg["features"] is None else f"[0, {rg['features']:.4g}]"
                n = "n/a" if rg["neck"] is None else f"[0, {rg['neck']:.4g}]"
                cap[meth].append(f"  {ROWS[k].replace(chr(10), ' '):30s} features {f}   neck {n}")
        print(f"[supp] {name} ({tag})", flush=True)
    for meth, p in pdfs.items():
        p.close()
        head = [f"Supplementary figure, {METHOD[meth]}: one page per image (12 val images: 9 seeded random draws + 3 "
                "miss cases, see selection.json), in selection order.",
                "Rows: Full model, Fusion baseline (CBAM + add), Thermal-only YOLO11n, RGB-only YOLO11n (seed 0, "
                "last.pt = EMA, epoch 12). Columns: RGB input | IR input | RGB features | Thermal features | Fused "
                "features | Neck output | Detections.",
                "Features = post-attention backbone features entering fusion (single-modality baselines: backbone "
                "layers 4/6/10 in their own modality's column); Fused = fusion output; Neck output = Detect inputs. "
                "Each map is the sum of the P3/P4/P5 maps, from the saved 1/4-resolution maps upsampled to 768x960.",
                ("Grad-CAM target: sum of person scores of raw pre-NMS predictions with conf>=0.25 and IoU>=0.5 to a GT "
                 "box; 'No qualifying prediction' = no such prediction, so Grad-CAM is undefined for that row."
                 if meth == "gradcam" else
                 "EigenCAM: first principal component of the centred activations, sign oriented to correlate "
                 "positively with the channel-mean activation, ReLU; gradient-free."),
                "Detections: NMS conf>=0.25, IoU 0.7 (red dashed); GT green.",
                "Colour scales per row: one shared by RGB / Thermal / Fused features, one for Neck output. "
                "Ranges [min, max]:", ""]
        open(os.path.join(OUT, f"supp_fig_{meth}_caption.txt"), "w").write("\n".join(head + cap[meth]) + "\n")
        print(f"[supp] {os.path.join(OUT, f'supp_fig_{meth}.pdf')}")


if __name__ == "__main__":
    main()
