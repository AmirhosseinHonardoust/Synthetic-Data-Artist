from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _empirical_cdf(x: np.ndarray) -> np.ndarray:
    ranks = x.argsort().argsort().astype(float) + 1.0
    return ranks / (len(x) + 1.0)


def _empirical_ppf(u: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Nearest-quantile inverse CDF for stability across NumPy versions."""
    return np.quantile(samples, u, method="nearest")


def fit_copula(df_num: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit a Gaussian copula over numeric columns."""
    X = df_num.to_numpy()
    U = np.column_stack([_empirical_cdf(X[:, j]) for j in range(X.shape[1])])
    Z = norm.ppf(U)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    corr = np.corrcoef(Z, rowvar=False)

    eps = 1e-6
    corr = (1 - eps) * corr + eps * np.eye(corr.shape[0])
    chol = np.linalg.cholesky(corr)
    return {"chol": chol, "samples": X}


def sample_copula(model: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Sample numeric rows from a fitted Gaussian copula."""
    p = model["chol"].shape[0]
    z = np.random.randn(n, p) @ model["chol"].T
    u = norm.cdf(z)
    samples = model["samples"]
    cols = [_empirical_ppf(u[:, j], samples[:, j]) for j in range(p)]
    return np.column_stack(cols)


def generate_copula(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataframe using Gaussian Copula + categorical sampling."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=range(n_rows))

    if numeric_cols:
        model = fit_copula(df[numeric_cols])
        out[numeric_cols] = sample_copula(model, n_rows)

    for col in categorical_cols:
        probs = df[col].value_counts(normalize=True)
        out[col] = rng.choice(probs.index.to_numpy(), size=n_rows, p=probs.to_numpy())

    return out[df.columns]
