from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


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

        categorical_rates[col] = (
            float((~synthetic.isin(allowed)).mean()) if len(synthetic) else 0.0
        )

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
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import pairwise_distances
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols = [
        col
        for col in numeric_cols + categorical_cols
        if col in df_real.columns and col in df_syn.columns
    ]

    if not cols or len(df_real) == 0 or len(df_syn) == 0:
        return {
            "privacy_note": "No comparable columns available for nearest-neighbor privacy proxy metrics.",
            "exact_duplicate_rate": None,
            "nearest_neighbor_distance_mean": None,
            "nearest_neighbor_distance_p05": None,
            "nearest_neighbor_distance_min": None,
        }

    real_sample = df_real[cols].sample(
        n=min(max_rows, len(df_real)),
        random_state=seed,
    ).copy()

    syn_sample = df_syn[cols].sample(
        n=min(max_rows, len(df_syn)),
        random_state=seed,
    ).copy()

    real_rows = set(real_sample.astype(str).agg("||".join, axis=1))
    syn_rows = syn_sample.astype(str).agg("||".join, axis=1)

    exact_duplicate_rate = float(syn_rows.isin(real_rows).mean())

    used_numeric = [col for col in numeric_cols if col in cols]
    used_categorical = [col for col in categorical_cols if col in cols]

    transformers = []

    if used_numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                used_numeric,
            )
        )

    if used_categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                used_categorical,
            )
        )

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

    This compares:

    1. Train on real training data, test on held-out real data.
    2. Train on synthetic data, test on the same held-out real data.

    The preprocessing is inside sklearn Pipelines so the feature dimensions stay
    consistent. This avoids train/test one-hot-encoding mismatches.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not target_col:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": "Target column not configured.",
        }

    if target_col not in df_real.columns:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": f"Target column {target_col!r} is missing from real data.",
        }

    if target_col not in df_syn.columns:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": f"Target column {target_col!r} is missing from synthetic data.",
        }

    real = df_real.dropna(subset=[target_col]).copy()
    synthetic = df_syn.dropna(subset=[target_col]).copy()

    if len(real) < 20:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": "Not enough real rows for train/test utility evaluation.",
        }

    feature_cols = [
        col
        for col in numeric_cols + categorical_cols
        if col != target_col and col in real.columns and col in synthetic.columns
    ]

    if not feature_cols:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": "No comparable feature columns available.",
        }

    X_real = real[feature_cols]
    y_real = real[target_col]

    X_syn = synthetic[feature_cols]
    y_syn = synthetic[target_col]

    used_numeric = [col for col in numeric_cols if col in feature_cols]
    used_categorical = [col for col in categorical_cols if col in feature_cols]

    is_classification = (
        y_real.dtype == "object"
        or str(y_real.dtype).startswith("category")
        or y_real.nunique(dropna=True) <= 20
    )

    if is_classification:
        if y_real.nunique(dropna=True) < 2:
            return {
                "ml_utility_available": False,
                "ml_utility_reason": "Classification target has fewer than two classes in real data.",
            }

        if y_syn.nunique(dropna=True) < 2:
            return {
                "ml_utility_available": False,
                "ml_utility_reason": "Classification target has fewer than two classes in synthetic data.",
            }

        class_counts = y_real.value_counts(dropna=True)
        can_stratify = bool((class_counts >= 2).all())

        stratify = y_real if can_stratify else None
    else:
        stratify = None

    try:
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real,
            y_real,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real,
            y_real,
            test_size=test_size,
            random_state=seed,
        )

    def make_preprocessor() -> ColumnTransformer:
        transformers = []

        if used_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    used_numeric,
                )
            )

        if used_categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", _make_one_hot_encoder()),
                        ]
                    ),
                    used_categorical,
                )
            )

        return ColumnTransformer(transformers=transformers, remainder="drop")

    if is_classification:
        estimator_factory = lambda: RandomForestClassifier(
            n_estimators=80,
            random_state=seed,
            min_samples_leaf=2,
            class_weight="balanced",
        )
    else:
        estimator_factory = lambda: RandomForestRegressor(
            n_estimators=80,
            random_state=seed,
            min_samples_leaf=2,
        )

    def make_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", estimator_factory()),
            ]
        )

    try:
        real_pipeline = make_pipeline()
        real_pipeline.fit(X_train_real, y_train_real)
        real_pred = real_pipeline.predict(X_test_real)

        synthetic_pipeline = make_pipeline()
        synthetic_pipeline.fit(X_syn, y_syn)
        synthetic_pred = synthetic_pipeline.predict(X_test_real)

        results: dict[str, Any] = {
            "ml_utility_available": True,
            "ml_utility_target": target_col,
            "ml_utility_task": "classification" if is_classification else "regression",
            "ml_utility_train_real_rows": int(len(X_train_real)),
            "ml_utility_train_synthetic_rows": int(len(X_syn)),
            "ml_utility_test_rows": int(len(X_test_real)),
        }

        if is_classification:
            real_accuracy = accuracy_score(y_test_real, real_pred)
            synthetic_accuracy = accuracy_score(y_test_real, synthetic_pred)

            real_f1 = f1_score(
                y_test_real,
                real_pred,
                average="macro",
                zero_division=0,
            )

            synthetic_f1 = f1_score(
                y_test_real,
                synthetic_pred,
                average="macro",
                zero_division=0,
            )

            results.update(
                {
                    "ml_utility_metric": "accuracy",
                    "train_real_test_real": float(real_accuracy),
                    "train_synthetic_test_real": float(synthetic_accuracy),
                    "real_model_accuracy": float(real_accuracy),
                    "synthetic_model_accuracy": float(synthetic_accuracy),
                    "real_model_f1_macro": float(real_f1),
                    "synthetic_model_f1_macro": float(synthetic_f1),
                    "utility_ratio": (
                        float(synthetic_accuracy / real_accuracy)
                        if not np.isclose(real_accuracy, 0.0)
                        else None
                    ),
                }
            )

            if y_test_real.nunique(dropna=True) == 2:
                try:
                    real_proba = real_pipeline.predict_proba(X_test_real)[:, 1]
                    synthetic_proba = synthetic_pipeline.predict_proba(X_test_real)[:, 1]

                    results["real_model_roc_auc"] = float(
                        roc_auc_score(y_test_real, real_proba)
                    )
                    results["synthetic_model_roc_auc"] = float(
                        roc_auc_score(y_test_real, synthetic_proba)
                    )
                except Exception:
                    pass

        else:
            real_r2 = r2_score(y_test_real, real_pred)
            synthetic_r2 = r2_score(y_test_real, synthetic_pred)

            real_mae = mean_absolute_error(y_test_real, real_pred)
            synthetic_mae = mean_absolute_error(y_test_real, synthetic_pred)

            results.update(
                {
                    "ml_utility_metric": "r2",
                    "train_real_test_real": float(real_r2),
                    "train_synthetic_test_real": float(synthetic_r2),
                    "real_model_r2": float(real_r2),
                    "synthetic_model_r2": float(synthetic_r2),
                    "real_model_mae": float(real_mae),
                    "synthetic_model_mae": float(synthetic_mae),
                    "utility_ratio": (
                        float(synthetic_r2 / real_r2)
                        if not np.isclose(real_r2, 0.0)
                        else None
                    ),
                }
            )

        return results

    except Exception as exc:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": f"ML utility evaluation failed: {exc}",
        }
