from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def js_distance(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Compute Jensen-Shannon distance between two numeric vectors after scaling."""
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    pa, _ = np.histogram(a, bins=bins, range=(0, 1), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    return float(jensenshannon(pa, pb))


def distribution_overlap_scores(df_real: pd.DataFrame, df_syn: pd.DataFrame, bins: int = 30) -> dict[str, float]:
    """Return 1 - JSD for each numeric column."""
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    scores: dict[str, float] = {}
    for col in num_cols:
        real = df_real[col].dropna().to_numpy()
        syn = df_syn[col].dropna().to_numpy()
        scores[col] = 1.0 - js_distance(real, syn, bins=bins)
    return scores


def correlation_diff_mean(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> float | None:
    """Mean absolute difference between numeric correlation matrices."""
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols:
        return None
    corr_r = df_real[num_cols].corr().to_numpy()
    corr_s = df_syn[num_cols].corr().to_numpy()
    return float(np.abs(corr_r - corr_s).mean())
