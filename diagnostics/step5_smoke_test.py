"""
Pre-grid smoke test: does the REAL dual-branch RGBT graph actually build and
run for a quantum config, or was the Ctrl-C'd run's stock-looking model
summary (182 layers, 2.59M params, single [3,16,3,2] stem) evidence of a
silent RGB-only fallback?

Builds the model through the EXACT SAME path run_step5_retrain.py uses --
train_ablation_study.ABLATIONS + make_trainer, unmodified, unchanged from
what the real grid would build. No science source is edited; every check
here is a runtime monkeypatch inside this throwaway script.

Two runs, epochs=1, fraction=0.02 (a ~40-image subset), corrected validator
in loop:
  1. cbam_rgvqf_proposed -- full six-signal check (model type, CBAM count,
     RGVQFFusion instance count, RGVQFFusion.forward calls, QNode calls,
     forward-input structure), plus a finite-loss check.
  2. cbam_add -- the config whose summary triggered the alarm. Reduced
     check: forward-input structure + CBAM count only (no RGVQFFusion/QNode
     -- fusion_mode="add" doesn't use them at all, so those two signals are
     N/A for this config, not a failure).

Writes diagnostics/step5_smoke.md and stops. Does not launch the grid.
"""
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import ultralytics
from ultralytics import YOLO

from phase3.code.train_ablation_study import ABLATIONS, make_trainer
from phase3.code.model.yolov11_rgbt import RGBTDetectionModel
from phase3.code.model.fusion.rgvqf import RGVQFFusion
import phase3.code.model.fusion.rgvqf as rgvqf_module

print("ultralytics version:", ultralytics.__version__)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_YAML = "phase3/dataset_aligned/data.yaml"
data_path = os.path.abspath(DATA_YAML)
with open(data_path) as f:
    data_content = f.read()
assert "ir:" in data_content, f"{DATA_YAML} is missing the ir: key -- not the live config"
print(f"[guardrail] live data config: {DATA_YAML}")
print(data_content)

OUT_DIR = "diagnostics/step5_smoke_runs"

# ---------------------------------------------------------------------------
# Shared instrumentation state (reset between the two runs)
# ---------------------------------------------------------------------------
state = {
    "model_forward_calls": [],   # first 3 calls: structure of the image input
    "rgvqf_forward_calls": 0,
    "rgvqf_first_shapes": None,
    "qnode_calls": 0,
    "losses": [],                # (step, finite: bool, value)
}


def _describe(x):
    if isinstance(x, dict):
        img = x.get("img")
        inner = _describe(img)
        inner["via_dict"] = True
        return inner
    if isinstance(x, (tuple, list)):
        return {
            "kind": "tuple/list",
            "len": len(x),
            "shapes": [tuple(t.shape) if hasattr(t, "shape") else type(t).__name__ for t in x],
            "via_dict": False,
        }
    if hasattr(x, "shape"):
        return {"kind": "single_tensor", "shape": tuple(x.shape), "via_dict": False}
    return {"kind": type(x).__name__, "via_dict": False}


_orig_model_forward = RGBTDetectionModel.forward


def _patched_model_forward(self, x, *args, **kwargs):
    if len(state["model_forward_calls"]) < 3:
        desc = _describe(x)
        state["model_forward_calls"].append(desc)
        print(f"[forward call #{len(state['model_forward_calls'])}] {desc}")
    return _orig_model_forward(self, x, *args, **kwargs)


_orig_rgvqf_forward = RGVQFFusion.forward


def _patched_rgvqf_forward(self, rgb, ir, w_rgb=0.5, w_th=0.5):
    state["rgvqf_forward_calls"] += 1
    out = _orig_rgvqf_forward(self, rgb, ir, w_rgb, w_th)
    if state["rgvqf_first_shapes"] is None:
        state["rgvqf_first_shapes"] = {
            "rgb_in": tuple(rgb.shape),
            "ir_in": tuple(ir.shape),
            "out": tuple(out.shape),
        }
        print(f"[RGVQFFusion.forward first call] {state['rgvqf_first_shapes']}")
    return out


_orig_qnode = rgvqf_module._rgvqf_circuit


def _patched_qnode(*args, **kwargs):
    state["qnode_calls"] += 1
    return _orig_qnode(*args, **kwargs)


def install_patches():
    RGBTDetectionModel.forward = _patched_model_forward
    RGVQFFusion.forward = _patched_rgvqf_forward
    rgvqf_module._rgvqf_circuit = _patched_qnode


def remove_patches():
    RGBTDetectionModel.forward = _orig_model_forward
    RGVQFFusion.forward = _orig_rgvqf_forward
    rgvqf_module._rgvqf_circuit = _orig_qnode


def reset_state():
    state["model_forward_calls"] = []
    state["rgvqf_forward_calls"] = 0
    state["rgvqf_first_shapes"] = None
    state["qnode_calls"] = 0
    state["losses"] = []


def on_train_batch_end(trainer):
    loss = getattr(trainer, "loss", None)
    if loss is None:
        return
    finite = bool(torch.isfinite(loss).all().item())
    val = loss.detach().float().sum().item()
    state["losses"].append((trainer.epoch, finite, val))
    print(f"[batch end] epoch={trainer.epoch} loss_finite={finite} loss_sum={val:.4f}")


def run_config(key, extra_name_suffix=""):
    reset_state()
    config = ABLATIONS[key]
    trainer_cls = make_trainer(config)
    print()
    print("=" * 70)
    print(f"RUN: {key} -> {config['name']}  fusion_mode={config['fusion_mode']}")
    print("=" * 70)

    model = YOLO("yolo11n.pt")
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.train(
        trainer=trainer_cls,
        data=DATA_YAML,
        epochs=1,
        imgsz=960,
        batch=4,
        device=device,
        workers=0,
        fraction=0.02,
        project=OUT_DIR,
        name=config["name"] + extra_name_suffix,
        exist_ok=True,
        patience=100,
        plots=False,
        val=True,
        rect=True,
        box=10.0, cls=0.3, dfl=1.5,
    )

    real_model = model.trainer.model
    model_type = type(real_model).__name__
    is_rgbt_model = isinstance(real_model, RGBTDetectionModel)

    cbam_count = sum(1 for m in real_model.modules() if "CBAM" in type(m).__name__)
    rgvqf_instance_count = sum(1 for m in real_model.modules() if isinstance(m, RGVQFFusion))

    all_losses_finite = all(f for _, f, _ in state["losses"]) and len(state["losses"]) > 0

    result = {
        "config_key": key,
        "config_name": config["name"],
        "fusion_mode": config["fusion_mode"],
        "model_type": model_type,
        "is_rgbt_model": is_rgbt_model,
        "cbam_count": cbam_count,
        "rgvqf_instance_count": rgvqf_instance_count,
        "rgvqf_forward_calls": state["rgvqf_forward_calls"],
        "rgvqf_first_shapes": state["rgvqf_first_shapes"],
        "qnode_calls": state["qnode_calls"],
        "forward_calls": list(state["model_forward_calls"]),
        "n_loss_steps": len(state["losses"]),
        "all_losses_finite": all_losses_finite,
    }
    print()
    print(f"RESULT[{key}]: {result}")
    return result


install_patches()
try:
    result_rgvqf = run_config("cbam_rgvqf")
    result_add = run_config("cbam_add")
finally:
    remove_patches()

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def paired_tuple_seen(result):
    for c in result["forward_calls"]:
        if c.get("kind") == "tuple/list" and c.get("len") == 2:
            return True
    return False


rgvqf_checks = {
    "model_is_RGBTDetectionModel": result_rgvqf["is_rgbt_model"],
    "cbam_count_gt_0": result_rgvqf["cbam_count"] > 0,
    "rgvqf_instance_count_gt_0": result_rgvqf["rgvqf_instance_count"] > 0,
    "rgvqf_forward_calls_gt_0": result_rgvqf["rgvqf_forward_calls"] > 0,
    "qnode_calls_gt_0": result_rgvqf["qnode_calls"] > 0,
    "forward_gets_paired_tuple": paired_tuple_seen(result_rgvqf),
    "losses_finite": result_rgvqf["all_losses_finite"],
}
rgvqf_pass = all(rgvqf_checks.values())

add_checks = {
    "forward_gets_paired_tuple": paired_tuple_seen(result_add),
    "cbam_count_gt_0": result_add["cbam_count"] > 0,
}
add_pass = all(add_checks.values())

overall_pass = rgvqf_pass and add_pass

lines = []
lines.append("# Step 5 pre-grid smoke test\n")
lines.append(f"ultralytics {ultralytics.__version__}, device={device}\n")
lines.append(f"Live data config: `{DATA_YAML}` (confirmed has `ir:` key)\n")
lines.append("\n## cbam_rgvqf_proposed (six signals)\n")
lines.append(f"- model type: `{result_rgvqf['model_type']}` "
             f"(is RGBTDetectionModel: {result_rgvqf['is_rgbt_model']}) -- **{'PASS' if rgvqf_checks['model_is_RGBTDetectionModel'] else 'FAIL'}**\n")
lines.append(f"- CBAM/CMAAM module count: {result_rgvqf['cbam_count']} -- "
             f"**{'PASS' if rgvqf_checks['cbam_count_gt_0'] else 'FAIL'}**\n")
lines.append(f"- RGVQFFusion instance count: {result_rgvqf['rgvqf_instance_count']} -- "
             f"**{'PASS' if rgvqf_checks['rgvqf_instance_count_gt_0'] else 'FAIL'}**\n")
lines.append(f"- RGVQFFusion.forward() calls: {result_rgvqf['rgvqf_forward_calls']} "
             f"(first call shapes: {result_rgvqf['rgvqf_first_shapes']}) -- "
             f"**{'PASS' if rgvqf_checks['rgvqf_forward_calls_gt_0'] else 'FAIL'}**\n")
lines.append(f"- QNode (`_rgvqf_circuit`) calls: {result_rgvqf['qnode_calls']} -- "
             f"**{'PASS' if rgvqf_checks['qnode_calls_gt_0'] else 'FAIL'}**\n")
lines.append(f"- forward-input structure (first 3 calls): {result_rgvqf['forward_calls']} -- "
             f"**{'PASS' if rgvqf_checks['forward_gets_paired_tuple'] else 'FAIL'}**\n")
lines.append(f"- loss finite over {result_rgvqf['n_loss_steps']} steps: {result_rgvqf['all_losses_finite']} -- "
             f"**{'PASS' if rgvqf_checks['losses_finite'] else 'FAIL'}**\n")
lines.append(f"\n**cbam_rgvqf_proposed verdict: {'PASS' if rgvqf_pass else 'FAIL'}**\n")

lines.append("\n## cbam_add (the config that triggered the alarm; reduced check)\n")
lines.append(f"- forward-input structure (first 3 calls): {result_add['forward_calls']} -- "
             f"**{'PASS' if add_checks['forward_gets_paired_tuple'] else 'FAIL'}**\n")
lines.append(f"- CBAM/CMAAM module count: {result_add['cbam_count']} -- "
             f"**{'PASS' if add_checks['cbam_count_gt_0'] else 'FAIL'}**\n")
lines.append(f"- (RGVQFFusion instances / QNode calls: N/A for fusion_mode=\"add\", not checked)\n")
lines.append(f"\n**cbam_add verdict: {'PASS' if add_pass else 'FAIL'}**\n")

lines.append(f"\n## Overall verdict: {'PASS -- grid is safe to launch' if overall_pass else 'FAIL -- do NOT launch'}\n")

report_path = "diagnostics/step5_smoke.md"
with open(report_path, "w") as f:
    f.writelines(lines)
print()
print(f"wrote {report_path}")
print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
