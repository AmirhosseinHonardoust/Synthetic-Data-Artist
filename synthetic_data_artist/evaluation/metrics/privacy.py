from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import _make_one_hot_encoder


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
            "privacy_note": (
                "No comparable columns available for nearest-neighbor privacy proxy " "metrics."
            ),
            "exact_duplicate_rate": None,
            "nearest_neighbor_distance_mean": None,
            "nearest_neighbor_distance_p05": None,
            "nearest_neighbor_distance_min": None,
        }

    real_sample = (
        df_real[cols]
        .sample(
            n=min(max_rows, len(df_real)),
            random_state=seed,
        )
        .copy()
    )

    syn_sample = (
        df_syn[cols]
        .sample(
            n=min(max_rows, len(df_syn)),
            random_state=seed,
        )
        .copy()
    )

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
        "privacy_note": (
            "Nearest-neighbor metrics are proxy diagnostics, not a formal privacy " "guarantee."
        ),
        "exact_duplicate_rate": exact_duplicate_rate,
        "nearest_neighbor_distance_mean": float(np.mean(nearest)),
        "nearest_neighbor_distance_p05": float(np.quantile(nearest, 0.05)),
        "nearest_neighbor_distance_min": float(np.min(nearest)),
    }
