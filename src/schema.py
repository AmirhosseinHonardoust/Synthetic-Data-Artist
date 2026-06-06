from __future__ import annotations

import pandas as pd


def detect_schema(df: pd.DataFrame, categorical_threshold: int = 20) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical columns using dtype plus low-cardinality integer logic."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            categorical_cols.append(col)
            continue

        numeric_cols.append(col)
        if df[col].nunique() <= categorical_threshold and not pd.api.types.is_float_dtype(df[col]):
            categorical_cols.append(col)
            numeric_cols.remove(col)

    return numeric_cols, categorical_cols
