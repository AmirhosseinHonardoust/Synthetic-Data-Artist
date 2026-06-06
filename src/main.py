from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:  # package imports, e.g. `python -m src.main`
    from .config import get_nested, load_config, save_json, set_seed
    from .data import load_or_generate
    from .evaluation.metrics import js_distance as _jsd
    from .evaluation.plots import (
        pairplot_compare,
        plot_correlation_heatmap,
        plot_distribution_overlap,
        plot_pca,
    )
    from .models.copula import (
        _empirical_cdf,
        _empirical_ppf,
        fit_copula,
        generate_copula,
        sample_copula,
    )
    from .models.vae import train_and_generate_vae
    from .reporting.html_report import write_report
    from .schema import detect_schema
except ImportError:  # script imports, e.g. `python src/main.py`
    from config import get_nested, load_config, save_json, set_seed
    from data import load_or_generate
    from evaluation.metrics import js_distance as _jsd
    from evaluation.plots import (
        pairplot_compare,
        plot_correlation_heatmap,
        plot_distribution_overlap,
        plot_pca,
    )
    from models.copula import (
        _empirical_cdf,
        _empirical_ppf,
        fit_copula,
        generate_copula,
        sample_copula,
    )
    from models.vae import train_and_generate_vae
    from reporting.html_report import write_report
    from schema import detect_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", choices=["copula", "vae"], default="copula")
    parser.add_argument("--data", default="data/real_data.csv")
    parser.add_argument("--run_name", default=None, help="Optional name to separate outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    rows_cfg = cfg.get("rows", None)
    bins = int(cfg.get("hist_bins", 30))
    pca_components = int(cfg.get("pca_components", 2))
    pairplot_sample = int(cfg.get("pairplot_sample", 500))
    cat_thr = int(cfg.get("categorical_threshold", 20))

    run_name = args.run_name or args.method
    data_dir = Path("data")
    run_dir = Path("outputs") / run_name
    plots_dir = run_dir / "plots"
    reports_dir = Path("reports")

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate(args.data, seed=seed)
    numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=cat_thr)
    n_rows = int(rows_cfg) if rows_cfg is not None else len(df)

    if args.method == "copula":
        df_syn = generate_copula(df, numeric_cols, categorical_cols, n_rows=n_rows, seed=seed)
    else:
        df_syn = train_and_generate_vae(
            df,
            numeric_cols,
            categorical_cols,
            n_rows=n_rows,
            seed=seed,
            epochs=int(get_nested(cfg, "vae.epochs", 30)),
            batch=int(get_nested(cfg, "vae.batch_size", 128)),
            latent=int(get_nested(cfg, "vae.latent_dim", 8)),
            hidden=int(get_nested(cfg, "vae.hidden_dim", 64)),
            learning_rate=float(get_nested(cfg, "vae.learning_rate", 1e-3)),
            kl_weight=float(get_nested(cfg, "vae.kl_weight", 1e-3)),
        )

    syn_path = data_dir / f"synthetic_data_{run_name}.csv"
    df_syn.to_csv(syn_path, index=False)

    dist_scores = plot_distribution_overlap(df, df_syn, bins=bins, out_path=plots_dir / "distribution_overlap.png")
    pca_info = plot_pca(df, df_syn, out_path=plots_dir / "pca_projection.png", n_components=pca_components)
    corr_info = plot_correlation_heatmap(df, df_syn, out_path=plots_dir / "correlation_heatmap.png")
    pairplot_compare(df, df_syn, out_path=plots_dir / "pairplot_comparison.png", sample=pairplot_sample)

    metrics = {
        "rows_real": int(len(df)),
        "rows_synthetic": int(len(df_syn)),
        "method": args.method,
        "seed": seed,
        "distribution_overlap_mean": float(np.mean(list(dist_scores.values()))) if dist_scores else None,
        "distribution_overlap_per_feature": dist_scores,
        "pca_explained_variance": (pca_info or {}).get("explained_variance"),
        **corr_info,
    }
    save_json(run_dir / "metrics.json", metrics)

    write_report(
        method=args.method,
        rows=n_rows,
        seed=seed,
        metrics=metrics,
        report_path=reports_dir / f"{run_name}_report.html",
        run_name=run_name,
    )

    print(f"Saved synthetic CSV: {syn_path}")
    print(f"Saved metrics:       {run_dir / 'metrics.json'}")
    print(f"Saved plots in:      {plots_dir}")
    print(f"Saved report:        {reports_dir / f'{run_name}_report.html'}")


if __name__ == "__main__":
    main()
