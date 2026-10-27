from .add import AddFusion
from .concat import ConcatFusion
from .qinspired import QInspiredFusion
from .rgvqf import RGVQFFusion

# "vqc" is kept as the primary key for backward compatibility: every
# existing trainer/script (rgbt_trainer.py, train.py, train_ablation_study.py)
# passes fusion_mode="vqc". "qinspired" is an alias for the same class,
# matching the advisor's renamed terminology going forward.
FUSION_REGISTRY = {
    "add": AddFusion,
    "concat": ConcatFusion,
    "vqc": QInspiredFusion,
    "qinspired": QInspiredFusion,
    "rgvqf": RGVQFFusion,
}


def build_fusion(mode, channels, **kwargs):
    if mode not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion mode {mode!r}, expected one of {sorted(FUSION_REGISTRY)}"
        )
    return FUSION_REGISTRY[mode](channels, **kwargs)
