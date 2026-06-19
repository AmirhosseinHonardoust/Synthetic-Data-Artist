from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when the input dataframe is not suitable for generation/evaluation."""


def validate_dataframe(df: pd.DataFrame, *, required_min_rows: int = 2) -> None:
    """Validate basic dataframe contracts before synthetic generation.

    This intentionally checks only universal contracts. Domain-specific checks
    should stay in downstream project configs or future schema validators.

    The function uses positional column access where needed so duplicate column
    names can be reported cleanly instead of causing pandas ambiguity errors.
    """
    errors: list[str] = []

    if df.empty:
        errors.append("dataframe is empty")

    if len(df) < required_min_rows:
        errors.append(f"dataframe must contain at least {required_min_rows} rows")

    if len(df.columns) == 0:
        errors.append("dataframe must contain at least one column")

    if df.columns.duplicated().any():
        duplicates = sorted({str(col) for col in df.columns[df.columns.duplicated(keep=False)]})
        errors.append(f"duplicate column names found: {duplicates}")

    blank_columns = [str(col) for col in df.columns if str(col).strip() == ""]
    if blank_columns:
        errors.append("column names must not be blank")

    all_null_columns: list[str] = []
    unsupported_columns: list[str] = []

    for idx, col in enumerate(df.columns):
        series = df.iloc[:, idx]

        if series.isna().all():
            all_null_columns.append(str(col))

        if pd.api.types.is_complex_dtype(series.dtype):
            unsupported_columns.append(str(col))

    if all_null_columns:
        errors.append(f"columns cannot be entirely missing: {all_null_columns}")

    if unsupported_columns:
        errors.append(f"unsupported complex-valued columns: {unsupported_columns}")

    if errors:
        raise DataValidationError("Invalid input data: " + "; ".join(errors))


def load_or_generate(
    csv_path: str | Path,
    seed: int = 42,
    generated_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the real dataset or generate a reproducible demo dataset when missing."""
    csv_path = Path(csv_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        validate_dataframe(df)
        return df

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1500,
        n_features=8,
        n_informative=5,
        random_state=seed,
    )

    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(8)])
    df["cat_a"] = pd.qcut(df["num_0"], q=4, labels=["A", "B", "C", "D"]).astype(str)
    df["cat_b"] = np.where(df["num_1"] > 0, "Yes", "No")
    df["target"] = y

    validate_dataframe(df)

    output_path = (
        Path(generated_output_path) if generated_output_path else Path("data/real_data.csv")
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)

    return df
