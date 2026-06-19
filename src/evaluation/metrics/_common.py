from __future__ import annotations

import numpy as np
import pandas as pd


def _as_clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _make_one_hot_encoder():
    """Create a OneHotEncoder compatible with newer and older sklearn versions."""
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
