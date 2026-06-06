from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .metrics import correlation_diff_mean, distribution_overlap_scores


def plot_distribution_overlap(df_real, df_syn, bins, out_path: Path):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols:
        return {}

    cols = num_cols[: min(6, len(num_cols))]
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 2.2 * len(cols)))
    if len(cols) == 1:
        axes = [axes]

    scores = distribution_overlap_scores(df_real[cols], df_syn[cols], bins=bins)
    for ax, col in zip(axes, cols):
        ax.hist(df_real[col].dropna(), bins=bins, alpha=0.5, label="real", density=True)
        ax.hist(df_syn[col].dropna(), bins=bins, alpha=0.5, label="synthetic", density=True)
        ax.set_title(f"Distribution overlap: {col}")
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return scores


def plot_pca(df_real, df_syn, out_path: Path, n_components=2):
    warnings.filterwarnings("ignore", message=".*feature names.*PCA.*")
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols:
        return None

    Xr = df_real[num_cols].fillna(df_real[num_cols].mean())
    Xs = df_syn[num_cols].fillna(df_syn[num_cols].mean())
    X = pd.concat([Xr, Xs], axis=0).to_numpy()
    pca = PCA(n_components=n_components, random_state=42).fit(X)
    Zr = pca.transform(Xr)
    Zs = pca.transform(Xs)

    plt.figure(figsize=(7, 5))
    plt.scatter(Zr[:, 0], Zr[:, 1], s=10, alpha=0.5, label="real")
    plt.scatter(Zs[:, 0], Zs[:, 1], s=10, alpha=0.5, label="synthetic")
    plt.title("PCA Projection: real vs synthetic")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {"explained_variance": pca.explained_variance_ratio_.tolist()}


def plot_correlation_heatmap(df_real, df_syn, out_path: Path):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols:
        return {}

    corr_r = df_real[num_cols].corr().to_numpy()
    corr_s = df_syn[num_cols].corr().to_numpy()
    diff = correlation_diff_mean(df_real, df_syn)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(corr_r, ax=axes[0], vmin=-1, vmax=1, cmap="coolwarm", cbar=False)
    axes[0].set_title("Real correlation")
    sns.heatmap(corr_s, ax=axes[1], vmin=-1, vmax=1, cmap="coolwarm", cbar=False)
    axes[1].set_title("Synthetic correlation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {"correlation_diff_mean": diff}


def pairplot_compare(df_real, df_syn, out_path: Path, sample=500):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if len(num_cols) < 2:
        return False

    cols = num_cols[: min(4, len(num_cols))]
    real = df_real[cols].sample(n=min(sample, len(df_real)), random_state=42).copy()
    real["__type__"] = "real"
    syn = df_syn[cols].sample(n=min(sample, len(df_syn)), random_state=42).copy()
    syn["__type__"] = "synthetic"
    combined = pd.concat([real, syn], axis=0)
    g = sns.pairplot(
        combined,
        vars=cols,
        hue="__type__",
        plot_kws={"alpha": 0.5, "s": 12},
        diag_kind="hist",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path)
    plt.close("all")
    return True
