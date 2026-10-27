#!/usr/bin/env python3
"""
Task 2: Grad-CAM / EigenCAM through the PAIRED (RGB, IR) path of campaign_e12 seed-0 models.

Read-only w.r.t. frozen code and runs; writes only into this folder.

  .venv/bin/python diagnostics/presubmission/gradcam/gradcam_paired.py select           # selection.json (no CAMs)
  .venv/bin/python diagnostics/presubmission/gradcam/gradcam_paired.py figures --n 1    # smoke test: first image only
  .venv/bin/python diagnostics/presubmission/gradcam/gradcam_paired.py figures          # all selected images + PDF
  .venv/bin/python diagnostics/presubmission/gradcam/gradcam_paired.py energy           # pointing-game over all val

Input = exactly what RGBTDetectionValidator feeds the model: YOLODataset(rect, pad=0, imgsz 960) -> 768x960 RGB
uint8 -> /255, and the validator's own IR code (cv2 gray, INTER_LINEAR resize, /255, repeat to 3ch).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
PRE = os.path.join(ROOT, "diagnostics/presubmission")
RUNS = os.path.join(PRE, "campaign_e12/runs")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import YOLODataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from phase3.code.rgbt_trainer import RGBTBatch, RGBTDetectionTrainer  # noqa: E402

MODELS = {  # row order in the figures
    "complete": "abl_complete_s0",
    "cbam_add": "abl_cbam_no_vqc_s0",
    "baseline_thermal": "base_yolo11n_thermal_s0",
    "baseline_rgb": "base_yolo11n_rgb_s0",
}
CONF, IOU_MATCH, NMS_IOU = 0.25, 0.5, 0.7
H, W = 768, 960
LEVELS = ("p3", "p4", "p5")
BASE_TAPS = {"p3": 4, "p4": 6, "p5": 10}  # backbone layers feeding the neck; 10 = last backbone stage
SEL_PATH = os.path.join(HERE, "selection.json")


# ------------------------------------------------------------------ data (validator-identical)
class ValData:
    def __init__(self, key):
        run = MODELS[key]
        args = yaml.safe_load(open(os.path.join(RUNS, run, "args.yaml")))
        self.fusion = not key.startswith("baseline")
        self.data = check_det_dataset(args["data"])
        hyp = get_cfg(overrides={k: v for k, v in args.items() if k not in ("save_dir",)})
        # identical to RGBTDetectionTrainer.build_dataset / BaselineTrainer.build_dataset (val, pad=0)
        self.ds = YOLODataset(img_path=self.data["val"], imgsz=args["imgsz"], batch_size=1, augment=False, hyp=hyp,
                              rect=True, cache=None, single_cls=False, stride=32, pad=0.0, prefix="val: ",
                              task="detect", classes=None, data=self.data, fraction=1.0)
        self.index = {os.path.basename(f): i for i, f in enumerate(self.ds.im_files)}
        if self.fusion:  # the trainer's own helpers, as RGBTDetectionValidator.preprocess uses them
            t = RGBTDetectionTrainer.__new__(RGBTDetectionTrainer)
            t.data, t.args = self.data, hyp
            self.ir_root = t._resolve_ir_root()

    def names(self):
        return sorted(self.index)

    def get(self, name):
        item = self.ds[self.index[name]]
        rgb = item["img"].float().unsqueeze(0) / 255.0  # Format() already gives RGB CHW
        assert tuple(rgb.shape[-2:]) == (H, W), f"expected 768x960, got {tuple(rgb.shape)}"
        b = item["bboxes"].clone()
        gt = xywh2xyxy(b) * torch.tensor([W, H, W, H])
        if not self.fusion:
            return rgb, None, gt
        ir_path = RGBTDetectionTrainer._matching_ir_path(self.ds.im_files[self.index[name]], self.ir_root)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        ir = cv2.resize(ir, (W, H), interpolation=cv2.INTER_LINEAR)
        ir = (torch.from_numpy(ir).float().unsqueeze(0) / 255.0).repeat(3, 1, 1).unsqueeze(0)
        return rgb, ir, gt


# ------------------------------------------------------------------ model + hooks
def load_model(key, device):
    ck = torch.load(os.path.join(RUNS, MODELS[key], "weights/last.pt"), map_location="cpu", weights_only=False)
    assert ck.get("epoch") == -1 and ck.get("ema") is None, "expected a finished, stripped last.pt (model == EMA)"
    m = ck["model"].float().eval().to(device)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


class Taps:
    """Collect activations (retain_grad) at the target layers for one forward pass."""

    def __init__(self, model, fusion):
        self.acts, self.hooks, self.fusion = {}, [], fusion
        self.paired_seen = False
        reg = self.hooks.append
        if fusion:
            fb = model.fusion_backbone

            def stock_called(*_):
                raise RuntimeError("stock single-stream backbone was called -- NOT the paired path")
            for i in range(11):
                reg(model.model[i].register_forward_pre_hook(stock_called))

            def top_pre(_, args):
                x = args[0]
                if not (isinstance(x, tuple) and len(x) == 2 and all(torch.is_tensor(t) for t in x)):
                    raise RuntimeError(f"model input is {type(x).__name__}, not a paired (RGB, IR) tuple -- abort")
                self.paired_seen = True
            reg(model.register_forward_pre_hook(top_pre))
            for lvl in LEVELS:
                fuse = getattr(fb, f"fuse_{lvl}")
                reg(fuse.register_forward_pre_hook(self._pre(lvl)))  # post-attention branch features
                reg(fuse.register_forward_hook(self._out(f"fused_{lvl}")))
        else:
            for lvl, i in BASE_TAPS.items():
                reg(model.model[i].register_forward_hook(self._out(f"bb_{lvl}")))
        reg(model.model[-1].register_forward_pre_hook(self._neck))

    def _keep(self, name, t):
        if t.requires_grad:
            t.retain_grad()
        self.acts[name] = t

    def _pre(self, lvl):
        def f(_, args):
            self._keep(f"rgb_{lvl}", args[0])
            self._keep(f"ir_{lvl}", args[1])
        return f

    def _out(self, name):
        return lambda _, __, out: self._keep(name, out)

    def _neck(self, _, args):
        for lvl, t in zip(LEVELS, args[0]):
            self._keep(f"neck_{lvl}", t)

    def remove(self):
        for h in self.hooks:
            h.remove()


def run_model(model, key, rgb, ir, gt, device):
    fusion = not key.startswith("baseline")
    taps = Taps(model, fusion)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if fusion:
                x = RGBTBatch((rgb.to(device), ir.to(device)))  # same container the validator passes
            else:
                x = (ir if key == "baseline_thermal" and ir is not None else rgb).to(device)
            out = model(x)
            y = out[0] if isinstance(out, (list, tuple)) else out  # eval Detect -> (y, raw)
            assert y.ndim == 3 and y.shape[1] == 5, f"unexpected head output {tuple(y.shape)}"
            if fusion:
                assert taps.paired_seen
            boxes, score = xywh2xyxy(y[0, :4].T.detach()), y[0, 4]
            with torch.no_grad():
                iou = box_iou(boxes.cpu(), gt) if len(gt) else torch.zeros(len(boxes), 0)
                best = iou.max(1).values if iou.shape[1] else torch.zeros(len(boxes))
                mask = ((score.detach().cpu() >= CONF) & (best >= IOU_MATCH)).to(device)
            target = (score * mask).sum()
            n_t = int(mask.sum())
            if n_t:
                target.backward()
        dets = non_max_suppression(y.detach(), conf_thres=CONF, iou_thres=NMS_IOU)[0].cpu()
        acts = {k: (v.detach()[0].float().cpu(), None if v.grad is None else v.grad[0].float().cpu())
                for k, v in taps.acts.items()}
    finally:
        taps.remove()
    return acts, dets, n_t, float(target.detach())


# ------------------------------------------------------------------ CAMs
def up(cam):
    return F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]


def gradcam(a, g):
    w = g.mean(dim=(1, 2))
    return up(torch.relu((w[:, None, None] * a).sum(0)))


def eigencam(a):
    c = a.reshape(a.shape[0], -1).T  # [hw, C]
    c = c - c.mean(0)
    _, _, vt = torch.linalg.svd(c, full_matrices=False)
    proj = (c @ vt[0]).reshape(a.shape[1:])
    if (proj * a.mean(0)).sum() < 0:  # SVD sign is arbitrary: orient with the channel-mean activation
        proj = -proj
    return up(torch.relu(proj))


def all_cams(acts, n_target):
    out = {"gradcam": {}, "eigencam": {}}
    for k, (a, g) in acts.items():
        out["eigencam"][k] = eigencam(a)
        if n_target and g is not None:
            out["gradcam"][k] = gradcam(a, g)
    for meth in out:  # level sums (raw, before any normalisation) for the figure columns
        for grp in ("rgb", "ir", "fused", "neck", "bb"):
            parts = [out[meth][f"{grp}_{l}"] for l in LEVELS if f"{grp}_{l}" in out[meth]]
            if len(parts) == 3:
                out[meth][f"{grp}_sum"] = sum(parts)
    return out


def box_mask(gt):
    m = torch.zeros(H, W, dtype=torch.bool)
    for x1, y1, x2, y2 in gt.round().int().tolist():
        m[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = True
    return m


# ------------------------------------------------------------------ selection (no CAMs involved)
def cmd_select(device):
    t0 = time.time()
    dc, dt = ValData("complete"), ValData("baseline_thermal")
    names = dc.names()
    assert names == dt.names() and len(names) == 1740
    mc, mt = load_model("complete", device), load_model("baseline_thermal", device)
    cand = []
    n_gt, n_found = 0, [0, 0]  # recall@0.25 sanity check of the input pipeline vs RESULTS.md
    with torch.no_grad():
        for i, n in enumerate(names):
            rgb, ir, gt = dc.get(n)
            if not len(gt):
                continue
            yc = mc(RGBTBatch((rgb.to(device), ir.to(device))))[0]
            yt = mt(dt.get(n)[0].to(device))[0]
            found = []
            for y in (yc, yt):
                d = non_max_suppression(y.float(), conf_thres=CONF, iou_thres=NMS_IOU)[0].cpu()
                iou = box_iou(gt, d[:, :4]) if len(d) else torch.zeros(len(gt), 0)
                found.append(iou.max(1).values >= IOU_MATCH if iou.shape[1] else torch.zeros(len(gt), dtype=torch.bool))
            n_gt += len(gt)
            n_found = [n_found[0] + int(found[0].sum()), n_found[1] + int(found[1].sum())]
            missed = (~found[0] & found[1]).nonzero().flatten().tolist()
            if missed:
                cand.append({"image": n, "gt_idx_missed_by_complete_found_by_thermal": missed})
            if i % 200 == 0:
                print(f"[select] {i}/1740  candidates={len(cand)}  {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    k_miss = min(3, len(cand))
    miss = [cand[j] for j in sorted(rng.choice(len(cand), k_miss, replace=False))] if k_miss else []
    miss_names = {c["image"] for c in miss}
    pool = [n for n in names if n not in miss_names]
    rand = [pool[j] for j in sorted(np.random.default_rng(0).choice(len(pool), 12 - k_miss, replace=False))]
    sel = {
        "rule": ("12 images total. (a) Miss cases: over all 1740 val images, a GT person counts as 'missed by complete, "
                 "found by thermal-only' if no NMS detection (conf>=0.25, NMS IoU 0.7) of complete has IoU>=0.5 with it "
                 "while one of baseline_thermal does; up to 3 images drawn from the candidate list with "
                 "numpy default_rng(0).choice. (b) Random: the remaining 12-k images drawn from the sorted val list "
                 "(miss images removed) with a fresh numpy default_rng(0).choice, no replacement. Fixed before any "
                 "heatmap was computed; uses detections only."),
        "models": MODELS, "conf": CONF, "iou_match": IOU_MATCH, "nms_iou": NMS_IOU,
        "n_val": len(names), "n_miss_candidates": len(cand), "miss_candidates": cand,
        "random": rand, "miss": miss, "images": rand + [c["image"] for c in miss],
        "recall_at_conf0.25_iou0.5": {"complete": n_found[0] / n_gt, "baseline_thermal": n_found[1] / n_gt,
                                      "n_gt": n_gt},
        "device": str(device), "time_s": round(time.time() - t0, 1),
    }
    json.dump(sel, open(SEL_PATH, "w"), indent=1)
    print(f"[select] {len(cand)} candidate images; selected {sel['images']}")


# ------------------------------------------------------------------ figures
def overlay(ax, img, cam, vmax, cmap):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.imshow(cam.numpy(), cmap=cmap, alpha=0.55, vmin=0, vmax=vmax, interpolation="bilinear")


def figure(name, results, meth, path_png, pdf=None):
    """Rows = models. Scale A (shared): RGB-branch, IR-branch, fused. Scale B: neck. Returns per-row ranges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cols = ["RGB input", "IR input", "RGB-branch CAM", "IR-branch CAM", "fused CAM", None, None, "neck -> Detect CAM",
            None, None, "detections (red) + GT (green)"]
    wr = [1, 1, 1, 1, 1, 0.04, 0.2, 1, 0.04, 0.2, 1]  # colorbar + spacer for tick labels after fused and after neck
    pw, ph = 2.6, 2.6 * H / W
    fig = plt.figure(figsize=(pw * sum(wr) * 1.06, ph * len(results) * 1.08 + 0.45))
    gs = fig.add_gridspec(len(results), len(cols), width_ratios=wr, wspace=0.05, hspace=0.06,
                          left=0.03, right=0.995, top=1 - 0.45 / (ph * len(results) * 1.08 + 0.45), bottom=0.005)
    cmap, ranges = "inferno", {}
    for r, (key, res) in enumerate(results.items()):
        rgb_img, ir_img, cams, gt, dets = res["rgb"], res["ir"], res["cams"][meth], res["gt"], res["dets"]
        fusion = not key.startswith("baseline")
        if fusion:
            branch = [cams.get("rgb_sum"), cams.get("ir_sum"), cams.get("fused_sum")]
        elif key == "baseline_rgb":
            branch = [cams.get("bb_sum"), None, None]
        else:
            branch = [None, cams.get("bb_sum"), None]
        neck = cams.get("neck_sum")
        va = max([float(p.max()) for p in branch if p is not None] + [1e-12])
        vb = float(neck.max()) if neck is not None else 1e-12
        ranges[key] = {"branch_fused": [0.0, va] if any(p is not None for p in branch) else None,
                       "neck": [0.0, vb] if neck is not None else None}
        used_rgb, used_ir = fusion or key == "baseline_rgb", fusion or key == "baseline_thermal"
        ax = [fig.add_subplot(gs[r, c]) for c in range(len(cols))]
        ax[0].imshow(rgb_img if used_rgb else rgb_img * 0.25)
        ax[1].imshow(ir_img if used_ir else ir_img * 0.25, cmap="gray", vmin=0, vmax=1)
        if not used_rgb:
            ax[0].text(0.5, 0.5, "not an input", ha="center", va="center", transform=ax[0].transAxes, fontsize=7, color="w")
        if not used_ir:
            ax[1].text(0.5, 0.5, "not an input", ha="center", va="center", transform=ax[1].transAxes, fontsize=7, color="w")
        for c in (6, 9):
            ax[c].axis("off")
        for c, p, bg, vmax in ((2, branch[0], rgb_img, va), (3, branch[1], ir_img, va), (4, branch[2], ir_img, va),
                               (7, neck, ir_img, vb)):
            if p is None:
                ax[c].set_facecolor("#dddddd")
                why = "no target preds" if (meth == "gradcam" and not res["n_target"]) else \
                    ("no fusion" if c == 4 else ("RGB not an input" if c == 2 else "IR not an input"))
                ax[c].text(0.5, 0.5, why, ha="center", va="center", transform=ax[c].transAxes, fontsize=7, color="#555")
            else:
                overlay(ax[c], bg, p, vmax, cmap)
        for c, vmax, have in ((5, va, any(p is not None for p in branch)), (8, vb, neck is not None)):
            if not have:
                ax[c].axis("off")
                continue
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax)), cax=ax[c])
            cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
            cb.ax.yaxis.get_offset_text().set_fontsize(4)
        a = ax[10]
        a.imshow(ir_img, cmap="gray", vmin=0, vmax=1)
        for x1, y1, x2, y2 in gt.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=0.7))
        for x1, y1, x2, y2, s, _ in dets.tolist():
            a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=0.7, ls="--"))
            a.text(x1, y1 - 4, f"{s:.2f}", color="red", fontsize=4)
        a.text(0.01, 0.99, f"target preds={res['n_target']}  sum={res['target']:.2f}", transform=a.transAxes,
               fontsize=5, va="top", color="white", bbox=dict(fc="black", alpha=0.5, lw=0, pad=1))
        ax[0].set_ylabel(key, fontsize=8)
        for c, t in enumerate(cols):
            if c not in (5, 6, 8, 9):
                ax[c].set_xticks([])
                ax[c].set_yticks([])
                if r == 0:
                    ax[c].set_title(t, fontsize=7, pad=3)
    title = {"gradcam": "Grad-CAM", "eigencam": "EigenCAM (gradient-free)"}[meth]
    fig.suptitle(f"{name} -- {title}. CAM = upsampled sum over P3/P4/P5. Per row: one scale for RGB-branch/IR-branch/"
                 f"fused, a separate scale for neck (ranges in caption file).", fontsize=7, y=0.999, va="top")
    fig.savefig(path_png, dpi=300)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return ranges


def write_caption(path, name, meth, ranges, res, sel):
    tag = "miss case (complete misses a GT person that thermal-only finds)" if name in {m["image"] for m in sel["miss"]} \
        else "random draw (seed 0)"
    lines = [f"{name} -- {'Grad-CAM' if meth == 'gradcam' else 'EigenCAM'} ({tag})",
             "Rows: complete, cbam_add, baseline_thermal, baseline_rgb (seed 0, last.pt = EMA, epoch 12). "
             "Input as validation: 768x960, RGB 3ch, IR 1ch replicated to 3ch, [0,1]; fusion models via the paired "
             "(RGB, IR) path.",
             "Columns: RGB input | IR input | RGB-branch CAM | IR-branch CAM | fused CAM | neck->Detect CAM | detections. "
             "Branch CAMs = post-attention features entering the fusion module; fused = fusion output; neck = Detect "
             "inputs; each the sum of P3/P4/P5 CAMs bilinearly upsampled to 768x960. Baselines: backbone layers 4/6/10 "
             "(10 = last backbone stage) in their own modality column.",
             ("Grad-CAM target: sum of person scores of raw pre-NMS predictions with conf>=0.25 and IoU>=0.5 to a GT box."
              if meth == "gradcam" else
              "EigenCAM: projection on the first principal component of the activations (centred), sign oriented to "
              "correlate positively with the channel-mean activation, ReLU. No gradients, no target."),
             "Detections: NMS conf>=0.25, IoU 0.7 (red dashed, with score); GT boxes green.",
             "Colour ranges per row [min, max]:"]
    fmt = lambda r: "n/a (Grad-CAM undefined: no target preds)" if r is None else f"[{r[0]:.3g}, {r[1]:.4g}]"  # noqa
    for k, rg in ranges.items():
        lines.append(f"  {k:17s} branch/fused {fmt(rg['branch_fused'])}   neck {fmt(rg['neck'])}   "
                     f"target preds={res[k]['n_target']}  "
                     f"dets={len(res[k]['dets'])}  GT={len(res[k]['gt'])}")
    open(path, "w").write("\n".join(lines) + "\n")


def figure_levels(name, results, meth, path_png):
    """Supplementary: every target layer separately (per-panel scale, since this is about location)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ["rgb", "ir", "fused", "neck"]
    cols = [f"{g}_{l}" for g in groups for l in LEVELS]
    fig, axes = plt.subplots(len(results), len(cols), figsize=(1.6 * len(cols), 1.45 * len(results)),
                             gridspec_kw=dict(wspace=0.03, hspace=0.1))
    for r, (key, res) in enumerate(results.items()):
        cams = res["cams"][meth]
        for c, col in enumerate(cols):
            k = col
            if key.startswith("baseline") and col.split("_")[0] in ("rgb", "ir"):
                mine = "rgb" if key == "baseline_rgb" else "ir"
                k = f"bb_{col.split('_')[1]}" if col.startswith(mine) else None
            ax = axes[r, c]
            bg = res["rgb"] if col.startswith("rgb") else res["ir"]
            if k and k in cams:
                ax.imshow(bg, cmap="gray" if bg.ndim == 2 else None)
                ax.imshow(cams[k].numpy(), cmap="inferno", alpha=0.55, vmin=0, vmax=float(cams[k].max()) + 1e-12)
            else:
                ax.set_facecolor("#dddddd")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col, fontsize=6)
        axes[r, 0].set_ylabel(key, fontsize=6)
    fig.suptitle(f"{name} -- {meth}, each target layer separately (per-panel scale)", fontsize=7)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_image(name, models, datas, device):
    results = {}
    for key, model in models.items():
        d = datas[key]
        if d.fusion:
            rgb, ir, gt = d.get(name)
            x_rgb, x_ir = rgb, ir
        else:
            x, _, gt = d.get(name)
            rgb, ir, _ = datas["complete"].get(name)  # display only
            x_rgb, x_ir = (x, None) if key == "baseline_rgb" else (rgb, x)
        acts, dets, n_t, tgt = run_model(model, key, x_rgb, x_ir, gt, device)
        results[key] = dict(rgb=rgb[0].permute(1, 2, 0).numpy(), ir=ir[0, 0].numpy(), gt=gt, dets=dets,
                            n_target=n_t, target=tgt, cams=all_cams(acts, n_t),
                            shapes={k: list(a.shape) for k, (a, _) in acts.items()})
    return results


def cmd_figures(device, n):
    from matplotlib.backends.backend_pdf import PdfPages
    sel = json.load(open(SEL_PATH))
    names = sel["images"][:n] if n else sel["images"]
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    datas = {k: ValData(k) for k in MODELS}
    models = {k: load_model(k, device) for k in MODELS}
    sfx = "all" if not n else f"smoke_{n}"
    pdfs = {m: PdfPages(os.path.join(out, f"{m}_{sfx}.pdf")) for m in ("gradcam", "eigencam")}
    meta = {}
    for name in names:
        t0 = time.time()
        res = process_image(name, models, datas, device)
        stem = os.path.splitext(name)[0]
        for meth in ("gradcam", "eigencam"):
            rng = figure(name, res, meth, os.path.join(out, f"{stem}_{meth}.png"), pdfs[meth])
            write_caption(os.path.join(out, f"{stem}_{meth}_caption.txt"), name, meth, rng, res, sel)
            figure_levels(name, res, meth, os.path.join(out, f"{stem}_{meth}_levels.png"))
        # CAM maps at 1/4 resolution (192x240), float16, compressed
        np.savez_compressed(os.path.join(out, f"{stem}_maps_q.npz"),
                            **{f"{k}__{m}__{l}": F.avg_pool2d(v[None, None], 4)[0, 0].half().numpy()
                               for k, r in res.items() for m, cs in r["cams"].items() for l, v in cs.items()})
        meta[name] = {k: dict(n_target=r["n_target"], target=r["target"], n_dets=len(r["dets"]), n_gt=len(r["gt"]),
                              act_shapes=r["shapes"]) for k, r in res.items()}
        print(f"[fig] {name} {time.time() - t0:.1f}s  " +
              "  ".join(f"{k}:t={r['n_target']}" for k, r in res.items()), flush=True)
    for p in pdfs.values():
        p.close()
    json.dump(meta, open(os.path.join(out, "figures_meta.json" if not n else "smoke_meta.json"), "w"), indent=1)


# ------------------------------------------------------------------ energy / pointing game over all val
def cmd_energy(device, limit):
    datas = {k: ValData(k) for k in MODELS}
    models = {k: load_model(k, device) for k in MODELS}
    names = datas["complete"].names()[:limit] if limit else datas["complete"].names()
    rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        for key, model in models.items():
            d = datas[key]
            if d.fusion:
                rgb, ir, gt = d.get(name)
            else:
                x, _, gt = d.get(name)
                rgb, ir = (x, None) if key == "baseline_rgb" else (None, x)
            if not len(gt):
                continue
            acts, _, n_t, _ = run_model(model, key, rgb, ir, gt, device)
            cams = all_cams(acts, n_t)
            m = box_mask(gt)
            area = float(m.float().mean())
            for meth, cs in cams.items():
                for layer, cam in cs.items():
                    tot = float(cam.sum())
                    rows.append(dict(image=name, model=key, method=meth, layer=layer, n_target=n_t, gt_area_frac=area,
                                     energy_in_gt=float(cam[m].sum()) / tot if tot > 0 else float("nan"),
                                     peak_in_gt=bool(m.flatten()[int(cam.argmax())]) if tot > 0 else None))
        if i % 50 == 0:
            print(f"[energy] {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
        if i % 100 == 0 or i == len(names) - 1:
            energy_summary(rows, final=i == len(names) - 1)


def energy_summary(rows, final):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "energy_per_image.csv"), index=False)
    if not final:
        return
    df["ratio"] = df["energy_in_gt"] / df["gt_area_frac"]
    s = (df.groupby(["method", "model", "layer"])
         .agg(n_images=("image", "nunique"), mean_energy_in_gt=("energy_in_gt", "mean"),
              sd_energy_in_gt=("energy_in_gt", "std"), mean_gt_area_frac=("gt_area_frac", "mean"),
              mean_per_image_ratio=("ratio", "mean"), pointing_game_acc=("peak_in_gt", "mean")).reset_index())
    s["ratio_of_means"] = s["mean_energy_in_gt"] / s["mean_gt_area_frac"]
    s.to_csv(os.path.join(HERE, "energy_summary.csv"), index=False)
    print(s.to_string(), flush=True)


def cmd_inputcheck(device, n=20):
    """Raw pre-NMS head output: this script's input path vs RGBTDetectionValidator/BaselineTrainer validator path,
    built exactly as run_final.revalidate builds it (trainer._setup_train -> test_loader -> get_validator)."""
    sys.path.insert(0, PRE)
    from torch.utils.data import DataLoader, Subset
    from run_final import run_settings, trainer_class
    from ultralytics import YOLO
    report = {}
    for key in ("complete", "baseline_thermal"):
        weights = os.path.join(RUNS, MODELS[key], "weights/last.pt")
        pre = YOLO(weights).model
        base_cls = trainer_class(key)

        class Reval(base_cls):
            def get_model(self, cfg=None, weights=None, verbose=True):
                return pre

        s = run_settings(key, 0, smoke=False)
        s.update(epochs=1, conf=0.001, iou=0.7, project=os.path.join(HERE, "_scratch"), name=f"inputcheck_{key}",
                 model="yolo11n.pt", device=str(device), plots=False)
        tr = Reval(overrides=s)
        tr._setup_train()
        v = tr.get_validator()
        v.device = tr.device
        ds = tr.test_loader.dataset
        idx = sorted(torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n].tolist())
        dl = DataLoader(Subset(ds, idx), batch_size=4, shuffle=False, collate_fn=ds.collate_fn)
        mine_data, mine_model = ValData(key), load_model(key, device)
        diffs, diffs_b1, in_diff, score_diff, names = [], [], [], [], []
        half = tr.device.type != "cpu" and bool(tr.amp)  # exactly how DetectionValidator.__call__ sets it
        v.args.half = half
        vm = (pre.half() if half else pre.float()).eval().to(device)
        for b in dl:
            x = v.preprocess(b)["img"]
            if key == "complete":
                assert isinstance(x, tuple) and len(x) == 2, "validator did not produce a paired tuple"
            with torch.no_grad():
                yv = vm(x)
                yv = (yv[0] if isinstance(yv, (list, tuple)) else yv).float().cpu()
            for j, f in enumerate(b["im_file"]):
                name = os.path.basename(f)
                rgb, ir, _ = mine_data.get(name)
                pair = key == "complete"
                with torch.no_grad():
                    y = mine_model(RGBTBatch((rgb.to(device), ir.to(device))) if pair else rgb.to(device))
                    y = (y[0] if isinstance(y, (list, tuple)) else y).float().cpu()[0]
                    xv1 = RGBTBatch((x[0][j:j + 1], x[1][j:j + 1])) if pair else x[j:j + 1]
                    y1 = vm(xv1)
                    y1 = (y1[0] if isinstance(y1, (list, tuple)) else y1).float().cpu()[0]
                assert y.shape == yv[j].shape, (y.shape, yv[j].shape)
                diffs.append(float((y - yv[j]).abs().max()))
                diffs_b1.append(float((y - y1).abs().max()))
                in_diff.append(max(float((rgb[0] - (x[0] if pair else x)[j].float().cpu()).abs().max()),
                                   float((ir[0] - x[1][j].float().cpu()).abs().max()) if pair else 0.0))
                score_diff.append(float((y[4] - yv[j][4]).abs().max()))
                names.append(name)
        report[key] = {"images": names, "pred_shape": list(y.shape), "validator_half": half, "trainer_amp": bool(tr.amp),
                       "max_abs_diff_input_tensors": max(in_diff),
                       "max_abs_diff_preds_same_batch_size_1": max(diffs_b1),
                       "max_abs_diff_preds_vs_validator_batch4": max(diffs),
                       "max_abs_diff_scores_vs_validator_batch4": max(score_diff),
                       "per_image_max_abs_diff_batch4": diffs,
                       "max_abs_pred_value": float(yv.abs().max())}
        print(f"[inputcheck] {key}: n={len(diffs)} input={max(in_diff):.1e} preds(b1)={max(diffs_b1):.1e} "
              f"preds(b4)={max(diffs):.1e} scores(b4)={max(score_diff):.1e} half={half}", flush=True)
        del tr, pre
    json.dump(report, open(os.path.join(HERE, "inputcheck.json"), "w"), indent=1)


def cmd_datacheck():
    import hashlib
    d = os.path.join(PRE, "data")
    layout = {"fusion": ("images/rgb", "labels/rgb"), "baseline_ir": ("images", "labels"),
              "baseline_rgb": ("images", "labels")}
    out = {}
    for split in ("train", "val"):
        stems, labels = {}, {}
        for ds, (im, lb) in layout.items():
            stems[ds] = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, ds, im, split))}
            lbdir = os.path.join(d, ds, lb, split)
            labels[ds] = {os.path.splitext(f)[0]: hashlib.sha256(open(os.path.join(lbdir, f), "rb").read()).hexdigest()
                          for f in os.listdir(lbdir) if f.endswith(".txt")}
        ref = stems["fusion"]
        out[split] = {
            "n_images": {k: len(v) for k, v in stems.items()},
            "image_stems_identical": all(v == ref for v in stems.values()),
            "n_label_files": {k: len(v) for k, v in labels.items()},
            "label_stems_identical": all(set(v) == set(labels["fusion"]) for v in labels.values()),
            "labels_byte_identical": all(v == labels["fusion"] for v in labels.values()),
            "images_without_label_file": len(ref - set(labels["fusion"])),
        }
    json.dump(out, open(os.path.join(HERE, "datacheck.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def pick_device(req):
    if req != "auto":
        return torch.device(req)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "figures", "energy", "inputcheck", "datacheck"])
    ap.add_argument("--n", type=int, default=0, help="figures: only the first n selected images (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="energy: first n val images only")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    print(f"[device] {dev}")
    {"select": lambda: cmd_select(dev), "figures": lambda: cmd_figures(dev, a.n),
     "energy": lambda: cmd_energy(dev, a.limit), "inputcheck": lambda: cmd_inputcheck(dev),
     "datacheck": cmd_datacheck}[a.cmd]()


if __name__ == "__main__":
    main()
