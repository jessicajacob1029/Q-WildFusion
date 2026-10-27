# Deprecated: VQCFusion moved to `model/fusion/qinspired.py` as
# `QInspiredFusion` (Stage 1 fusion registry refactor). This alias is kept so
# any existing external import of `VQCFusion` keeps working unchanged.
from .fusion.qinspired import QInspiredFusion as VQCFusion  # noqa: F401
