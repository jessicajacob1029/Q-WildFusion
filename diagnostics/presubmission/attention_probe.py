#!/usr/bin/env python3
"""
Channel/spatial attention after the B1 fix, on campaign_e12 last.pt (epoch 12). READ-ONLY, CPU, eval mode,
100 random val images (fixed subset, seed 0, same as A2), batch 1. Gates are recomputed from each attention
module's captured inputs with the module's own layers:
  CBAM : channel g = 2*sigmoid(mlp(avg x) + mlp(max x)); hidden pre-act = mlp[0](avg x), mlp[0](max x)
         spatial s = 2*sigmoid(conv7([mean_c x', max_c x'])), x' = x*g
  CMAAM: joint d = [avg r, max r, avg t, max t]; g_m = 2*sigmoid(mlp_m(d)); hidden pre-act = mlp_m[0](d)
         spatial s_m = 2*sigmoid(conv7_m([mean r', max r', mean t', max t']))
Writes campaign_e12/attention_probe.md / .json.
    .venv/bin/python -I diagnostics/presubmission/attention_probe.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from phase3.code.model.attention.cmaam import CMAAM  # noqa: E402
from phase3.code.model.cbam import CBAM  # noqa: E402

sys.argv = [sys.argv[0]]
import run_final as rf  # noqa: E402

C = os.path.join(HERE, "campaign_e12")
RUNS = [("complete", "abl_complete_s0"), ("cmaam_add", "abl_cmaam_add_s0"),
        ("cmaam_rgvqf_rem", "abl_cmaam_rgvqf_rem_s0"), ("cbam_add", "abl_cbam_no_vqc_s0")]


def stats_channel(g):  # g: [N, C]
    return dict(min=float(g.min()), max=float(g.max()), mean_abs_dev=float((g - 1).abs().mean()),
                std_across_images=float(g.std(0).mean()))


def stats_spatial(s):  # s: [N, H*W] (same H, W for all images: rect 768x960)
    return dict(min=float(s.min()), max=float(s.max()), mean_abs_dev=float((s - 1).abs().mean()),
                std_across_images=float(s.std(0).mean()))


def probe(key, run):
    tr = rf.trainer_class(key)(overrides=dict(rf.run_settings(key, 0, False), device="cpu", model="yolo11n.pt",
                                              project=os.path.join(HERE, "_attn_scratch"), name="attn"))
    m = YOLO(os.path.join(C, "runs", run, "weights", "last.pt")).model.float().eval()
    tr.model = m
    tr.set_model_attributes()
    store = {}

    def rec(name, kind, t):
        store.setdefault(name, {}).setdefault(kind, []).append(t.detach().float().cpu())

    def cbam_hook(name):
        def h(mod, inp):
            x = inp[0]
            with torch.no_grad():
                a, mx = mod.avgPool(x), mod.maxPool(x)
                rec(name, "preact", torch.cat([mod.mlp[0](a).flatten(1), mod.mlp[0](mx).flatten(1)], 1))
                g = 2 * torch.sigmoid(mod.mlp(a) + mod.mlp(mx))
                rec(name, "channel", g.flatten(1))
                xc = x * g
                s = 2 * torch.sigmoid(mod.spatialConv(torch.cat([xc.mean(1, keepdim=True), xc.max(1, keepdim=True)[0]], 1)))
                rec(name, "spatial", s.flatten(1))
        return h

    def cmaam_hook(name):
        def h(mod, inp):
            r, t = inp
            with torch.no_grad():
                d = torch.cat([mod.avg_pool(r), mod.max_pool(r), mod.avg_pool(t), mod.max_pool(t)], 1)
                gr, gt = 2 * torch.sigmoid(mod.channel_mlp_rgb(d)), 2 * torch.sigmoid(mod.channel_mlp_ir(d))
                rec(name + ".rgb", "preact", mod.channel_mlp_rgb[0](d).flatten(1))
                rec(name + ".ir", "preact", mod.channel_mlp_ir[0](d).flatten(1))
                rec(name + ".rgb", "channel", gr.flatten(1))
                rec(name + ".ir", "channel", gt.flatten(1))
                r2, t2 = r * gr, t * gt
                js = torch.cat([r2.mean(1, keepdim=True), r2.max(1, keepdim=True)[0], t2.mean(1, keepdim=True),
                                t2.max(1, keepdim=True)[0]], 1)
                rec(name + ".rgb", "spatial", (2 * torch.sigmoid(mod.spatial_conv_rgb(js))).flatten(1))
                rec(name + ".ir", "spatial", (2 * torch.sigmoid(mod.spatial_conv_ir(js))).flatten(1))
        return h

    acts = set()
    for name, mod in m.fusion_backbone.named_modules():
        if isinstance(mod, CBAM):
            mod.register_forward_pre_hook(cbam_hook(name))
            acts.add(type(mod.mlp[1]).__name__)
        elif isinstance(mod, CMAAM):
            mod.register_forward_pre_hook(cmaam_hook(name))
            acts.add(type(mod.channel_mlp_rgb[1]).__name__)
    ds = tr.build_dataset(tr.data["val"], mode="val", batch=1)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:100].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, collate_fn=ds.collate_fn)
    tr.test_loader = dl
    v = tr.get_validator()
    v.device = tr.device
    v.args.half = False
    with torch.no_grad():
        for b in dl:
            m(v.preprocess(b)["img"])
    out = {"activation": sorted(acts)}
    for name, d in store.items():
        ch, sp, pa = torch.cat(d["channel"]), torch.cat(d["spatial"]), torch.cat(d["preact"])
        out[name] = dict(channel=stats_channel(ch), spatial=stats_spatial(sp),
                         preact_pos_frac=float((pa > 0).float().mean()), n_images=int(ch.shape[0]))
    return out


def main():
    res = {}
    L = ["# Channel vs spatial attention after the B1 fix (campaign_e12, last.pt epoch 12, seed 0)\n\n"
         "100 random val images (fixed subset, seed 0), eval mode, batch 1. Gates are 2·sigmoid(·), so 1.0 = identity. "
         "\"std across images\" = per-channel (or per-pixel) std over the 100 images, averaged over channels (pixels).\n\n"]
    for key, run in RUNS:
        r = probe(key, run)
        res[run] = r
        L.append(f"## {run} (`{key}`; hidden activation {', '.join(r['activation'])})\n\n"
                 "| module · level · modality | channel min / max | channel mean \\|g−1\\| | channel std across images | "
                 "hidden pre-act > 0 | spatial min / max | spatial mean \\|s−1\\| | spatial std across images |\n"
                 "|---|---|---|---|---|---|---|---|\n")
        for name in sorted(k for k in r if k != "activation"):
            x = r[name]
            c, s = x["channel"], x["spatial"]
            L.append(f"| {name} | {c['min']:.3f} / {c['max']:.3f} | {c['mean_abs_dev']:.4f} | {c['std_across_images']:.4f} | "
                     f"{x['preact_pos_frac']:.3f} | {s['min']:.3f} / {s['max']:.3f} | {s['mean_abs_dev']:.4f} | "
                     f"{s['std_across_images']:.4f} |\n")
        L.append("\n")
    json.dump(res, open(os.path.join(C, "attention_probe.json"), "w"), indent=1)
    open(os.path.join(C, "attention_probe.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
