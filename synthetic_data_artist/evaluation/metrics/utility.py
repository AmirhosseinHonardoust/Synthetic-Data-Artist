from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import _make_one_hot_encoder


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
                "ml_utility_reason": (
                    "Classification target has fewer than two classes in real data."
                ),
            }

        if y_syn.nunique(dropna=True) < 2:
            return {
                "ml_utility_available": False,
                "ml_utility_reason": (
                    "Classification target has fewer than two classes in synthetic data."
                ),
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

        def estimator_factory() -> Any:
            return RandomForestClassifier(
                n_estimators=80,
                random_state=seed,
                min_samples_leaf=2,
                class_weight="balanced",
            )

    else:

        def estimator_factory() -> Any:
            return RandomForestRegressor(
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

                    results["real_model_roc_auc"] = float(roc_auc_score(y_test_real, real_proba))
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
                        float(synthetic_r2 / real_r2) if not np.isclose(real_r2, 0.0) else None
                    ),
                }
            )

        return results

    except Exception as exc:
        return {
            "ml_utility_available": False,
            "ml_utility_reason": f"ML utility evaluation failed: {exc}",
        }
