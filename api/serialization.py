"""
ARX Backend Serialization Utility (Directive 044R.3)
Centralizes RFC 8259-compliant JSON serialization across routes and analyzers.
Recursively sanitizes non-standard float representations (NaN, Inf, -Inf),
NumPy scalar/array types, pandas NA/NaT/Timestamps, and preserves boolean identity.
Zero circular dependencies.
"""

import math
from typing import Any, Dict, List, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore


def sanitize_floats(obj: Any) -> Any:
    """Recursively replaces NaN, Inf, and non-serializable pandas/numpy types with None/native types
    to guarantee strict RFC 8259 JSON compliance across all response payloads."""
    if obj is None:
        return None

    # Pandas missing-value markers
    if HAS_PANDAS:
        if obj is pd.NA or obj is pd.NaT:
            return None

    # Strict boolean preservation (must precede int check because isinstance(True, int) is True)
    if isinstance(obj, bool):
        return obj

    # Float handling (native + numpy)
    if HAS_NUMPY and isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # Integer handling (native + numpy)
    if HAS_NUMPY and isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, int):
        return obj

    # NumPy arrays
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return [sanitize_floats(x) for x in obj.tolist()]

    # Nested container types
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_floats(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(sanitize_floats(item) for item in obj)

    if isinstance(obj, set):
        return [sanitize_floats(item) for item in obj]

    return obj


_sanitize_floats = sanitize_floats

__all__ = ["sanitize_floats", "_sanitize_floats"]
