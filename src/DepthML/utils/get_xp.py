from typing import Any

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None


def get_xp(data: Any) -> Any:
    if cp is not None and isinstance(data, (cp.ndarray, cp.generic)):
        return cp
    return np
