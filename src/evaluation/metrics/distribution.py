from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ._common import _as_clean_numeric


def js_distance(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Compute Jensen-Shannon distance between two numeric vectors after min-max scaling."""
    a = _as_clean_numeric(a)
    b = _as_clean_numeric(b)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    min_v = float(min(a.min(), b.min()))
    max_v = float(max(a.max(), b.max()))

    if np.isclose(max_v, min_v):
        return 0.0

    a_scaled = (a - min_v) / (max_v - min_v + 1e-12)
    b_scaled = (b - min_v) / (max_v - min_v + 1e-12)

    pa, _ = np.histogram(a_scaled, bins=bins, range=(0, 1), density=False)
    pb, _ = np.histogram(b_scaled, bins=bins, range=(0, 1), density=False)

    pa = pa.astype(float) / (pa.sum() + 1e-12)
    pb = pb.astype(float) / (pb.sum() + 1e-12)

    return float(jensenshannon(pa, pb))


def distribution_overlap_scores(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    bins: int = 30,
    numeric_cols: list[str] | None = None,
) -> dict[str, float | None]:
    """Return 1 - Jensen-Shannon distance for each numeric column."""
    cols = numeric_cols or [
        col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col])
    ]

    scores: dict[str, float | None] = {}

    for col in cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue

        dist = js_distance(df_real[col].to_numpy(), df_syn[col].to_numpy(), bins=bins)
        scores[col] = None if np.isnan(dist) else float(1.0 - dist)

    return scores


def correlation_diff_mean(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> float | None:
    """Mean absolute difference between numeric correlation matrices."""
    cols = numeric_cols or [
        col for col in df_real.columns if pd.api.types.is_numeric_dtype(df_real[col])
    ]

    cols = [col for col in cols if col in df_real.columns and col in df_syn.columns]

    if len(cols) < 2:
        return None

    corr_real = df_real[cols].corr().fillna(0.0).to_numpy()
    corr_syn = df_syn[cols].corr().fillna(0.0).to_numpy()

    return float(np.abs(corr_real - corr_syn).mean())


def categorical_distribution_similarity(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Compare categorical distributions using total-variation similarity.

    For each categorical column, the score is 1 - total variation distance.
    A score near 1 means the synthetic category proportions are close to real data.
    """
    per_feature: dict[str, float] = {}

    for col in categorical_cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue

        real_counts = df_real[col].astype(str).value_counts(normalize=True)
        syn_counts = df_syn[col].astype(str).value_counts(normalize=True)

        categories = sorted(set(real_counts.index) | set(syn_counts.index))

        real = real_counts.reindex(categories, fill_value=0.0).to_numpy()
        syn = syn_counts.reindex(categories, fill_value=0.0).to_numpy()

        total_variation = 0.5 * np.abs(real - syn).sum()
        per_feature[col] = float(max(0.0, 1.0 - total_variation))

    return {
        "categorical_similarity_mean": (
            float(np.mean(list(per_feature.values()))) if per_feature else None
        ),
        "categorical_similarity_per_feature": per_feature,
    }


def numeric_summary_differences(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, Any]:
    """Compute normalized summary-statistic differences for numeric columns."""
    per_feature: dict[str, dict[str, float]] = {}
    flat_diffs: list[float] = []

    for col in numeric_cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue

        real = _as_clean_numeric(df_real[col])
        syn = _as_clean_numeric(df_syn[col])

        if len(real) == 0 or len(syn) == 0:
            continue

        scale = float(np.std(real) or 1.0)

        if np.isclose(scale, 0.0):
            scale = max(abs(float(np.mean(real))), 1.0)

        diffs = {
            "mean_abs_diff_scaled": abs(float(np.mean(real)) - float(np.mean(syn))) / scale,
            "std_abs_diff_scaled": abs(float(np.std(real)) - float(np.std(syn))) / scale,
            "min_abs_diff_scaled": abs(float(np.min(real)) - float(np.min(syn))) / scale,
            "max_abs_diff_scaled": abs(float(np.max(real)) - float(np.max(syn))) / scale,
        }

        per_feature[col] = diffs
        flat_diffs.extend(diffs.values())

    return {
        "numeric_summary_diff_mean": float(np.mean(flat_diffs)) if flat_diffs else None,
        "numeric_summary_diff_per_feature": per_feature,
    }


def boundary_violation_rates(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Check whether synthetic values violate simple real-data boundaries."""
    numeric_rates: dict[str, float] = {}
    categorical_rates: dict[str, float] = {}

    for col in numeric_cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue

        real = _as_clean_numeric(df_real[col])
        syn = _as_clean_numeric(df_syn[col])

        if len(real) == 0 or len(syn) == 0:
            continue

        lower = float(np.min(real))
        upper = float(np.max(real))

        numeric_rates[col] = float(((syn < lower) | (syn > upper)).mean())

    for col in categorical_cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue

        allowed = set(df_real[col].astype(str).dropna().unique())
        synthetic = df_syn[col].astype(str).dropna()

        categorical_rates[col] = float((~synthetic.isin(allowed)).mean()) if len(synthetic) else 0.0

    all_rates = list(numeric_rates.values()) + list(categorical_rates.values())

    return {
        "boundary_violation_rate_mean": float(np.mean(all_rates)) if all_rates else None,
        "numeric_boundary_violation_rate": numeric_rates,
        "categorical_invalid_rate": categorical_rates,
    }
