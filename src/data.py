from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_or_generate(csv_path: str | Path, seed: int = 42) -> pd.DataFrame:
    """Load the real dataset or generate a reproducible demo dataset when missing."""
    csv_path = Path(csv_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

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

    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/real_data.csv", index=False)
    return df
