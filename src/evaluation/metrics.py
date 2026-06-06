from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _as_clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


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
) -> dict[str, float]:
    """Return 1 - Jensen-Shannon distance for each numeric column."""
    cols = numeric_cols or [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    scores: dict[str, float] = {}
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
    cols = numeric_cols or [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if len(cols) < 2:
        return None
    corr_r = df_real[cols].corr().fillna(0.0).to_numpy()
    corr_s = df_syn[cols].corr().fillna(0.0).to_numpy()
    return float(np.abs(corr_r - corr_s).mean())


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
        "categorical_similarity_mean": float(np.mean(list(per_feature.values()))) if per_feature else None,
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
        lower, upper = float(np.min(real)), float(np.max(real))
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


def privacy_nearest_neighbor_metrics(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_rows: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute simple privacy-risk proxy metrics based on nearest real records.

    These metrics do not certify privacy. They are lightweight diagnostics:
    exact duplicate rate and nearest-neighbor distances from synthetic rows to real rows.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cols = [c for c in numeric_cols + categorical_cols if c in df_real.columns and c in df_syn.columns]
    if not cols or len(df_real) == 0 or len(df_syn) == 0:
        return {
            "privacy_note": "No comparable columns available for nearest-neighbor privacy proxy metrics.",
            "exact_duplicate_rate": None,
            "nearest_neighbor_distance_mean": None,
            "nearest_neighbor_distance_p05": None,
            "nearest_neighbor_distance_min": None,
        }

    real_sample = df_real[cols].sample(n=min(max_rows, len(df_real)), random_state=seed).copy()
    syn_sample = df_syn[cols].sample(n=min(max_rows, len(df_syn)), random_state=seed).copy()

    real_rows = set(real_sample.astype(str).agg("||".join, axis=1))
    syn_rows = syn_sample.astype(str).agg("||".join, axis=1)
    exact_duplicate_rate = float(syn_rows.isin(real_rows).mean())

    transformers = []
    used_numeric = [c for c in numeric_cols if c in cols]
    used_categorical = [c for c in categorical_cols if c in cols]
    if used_numeric:
        transformers.append(("num", StandardScaler(), used_numeric))
    if used_categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # pragma: no cover - older sklearn
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", encoder, used_categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_real = preprocessor.fit_transform(real_sample)
    X_syn = preprocessor.transform(syn_sample)
    distances = pairwise_distances(X_syn, X_real, metric="euclidean")
    nearest = distances.min(axis=1)

    return {
        "privacy_note": "Nearest-neighbor metrics are proxy diagnostics, not a formal privacy guarantee.",
        "exact_duplicate_rate": exact_duplicate_rate,
        "nearest_neighbor_distance_mean": float(np.mean(nearest)),
        "nearest_neighbor_distance_p05": float(np.quantile(nearest, 0.05)),
        "nearest_neighbor_distance_min": float(np.min(nearest)),
    }


def ml_utility_metrics(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
    test_size: float = 0.25,
) -> dict[str, Any]:
    """Estimate downstream utility with train-on-synthetic, test-on-real evaluation.

    Regression returns R2. Classification returns accuracy. This is intentionally
    lightweight and intended as a first diagnostic, not a complete utility study.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if not target_col or target_col not in df_real.columns or target_col not in df_syn.columns:
        return {"ml_utility_available": False, "ml_utility_reason": "Target column not configured or missing."}

    feature_cols = [c for c in numeric_cols + categorical_cols if c != target_col and c in df_real.columns and c in df_syn.columns]
    if not feature_cols:
        return {"ml_utility_available": False, "ml_utility_reason": "No comparable feature columns available."}

    X_real = df_real[feature_cols]
    y_real = df_real[target_col]
    X_syn = df_syn[feature_cols]
    y_syn = df_syn[target_col]

    stratify = y_real if not pd.api.types.is_numeric_dtype(y_real) and y_real.nunique() > 1 else None
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real,
        y_real,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    used_numeric = [c for c in numeric_cols if c in feature_cols]
    used_categorical = [c for c in categorical_cols if c in feature_cols]
    transformers = []
    if used_numeric:
        transformers.append(("num", StandardScaler(), used_numeric))
    if used_categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # pragma: no cover
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", encoder, used_categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    is_regression = pd.api.types.is_numeric_dtype(y_real) and y_real.nunique() > 10
    if is_regression:
        model = RandomForestRegressor(n_estimators=80, random_state=seed, min_samples_leaf=2)
        metric_name = "r2"
        scorer = r2_score
    else:
        model = RandomForestClassifier(n_estimators=80, random_state=seed, min_samples_leaf=2)
        metric_name = "accuracy"
        scorer = accuracy_score

    real_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    synthetic_pipeline = Pipeline([("preprocess", preprocessor), ("model", model.__class__(**model.get_params()))])

    real_pipeline.fit(X_train_real, y_train_real)
    synthetic_pipeline.fit(X_syn, y_syn)

    real_score = float(scorer(y_test_real, real_pipeline.predict(X_test_real)))
    synthetic_score = float(scorer(y_test_real, synthetic_pipeline.predict(X_test_real)))

    return {
        "ml_utility_available": True,
        "ml_utility_target": target_col,
        "ml_utility_metric": metric_name,
        "train_real_test_real": real_score,
        "train_synthetic_test_real": synthetic_score,
        "utility_ratio": float(synthetic_score / real_score) if not np.isclose(real_score, 0.0) else None,
    }
