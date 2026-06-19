from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_nested, load_config, save_json, set_seed, validate_config
from .data import load_or_generate, validate_dataframe
from .evaluation.metrics import (
    boundary_violation_rates,
    categorical_distribution_similarity,
    ml_utility_metrics,
    numeric_summary_differences,
    privacy_nearest_neighbor_metrics,
)
from .evaluation.plots import (
    pairplot_compare,
    plot_correlation_heatmap,
    plot_distribution_overlap,
    plot_pca,
)
from .models.copula import generate_copula
from .models.vae import train_and_generate_vae
from .reporting.html_report import write_report
from .schema import detect_schema


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate synthetic tabular data with Copula or VAE methods."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--method", choices=["copula", "vae"], default="copula")
    parser.add_argument("--data", default="data/real_data.csv", help="Path to real input CSV.")
    parser.add_argument("--run_name", default=None, help="Optional name to separate outputs.")
    parser.add_argument("--rows", type=int, default=None, help="Override number of synthetic rows.")
    parser.add_argument("--outdir", default=None, help="Root directory for run outputs.")
    parser.add_argument("--data-outdir", default=None, help="Directory for synthetic CSV outputs.")
    parser.add_argument("--report-dir", default=None, help="Directory for HTML reports.")
    parser.add_argument(
        "--skip-pairplot",
        action="store_true",
        help="Skip the pairplot. Useful for fast local checks and CI.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config/data/schema and exit without generating synthetic data.",
    )
    return parser.parse_args(argv)


def _resolve_path(cli_value: str | None, config: dict, key: str, default: str) -> Path:
    value = cli_value if cli_value is not None else get_nested(config, key, default)
    return Path(str(value))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    validate_config(cfg)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    rows_cfg = args.rows if args.rows is not None else cfg.get("rows", None)
    bins = int(cfg.get("hist_bins", 30))
    pca_components = int(cfg.get("pca_components", 2))
    pairplot_sample = int(cfg.get("pairplot_sample", 500))
    cat_thr = int(cfg.get("categorical_threshold", 20))

    if rows_cfg is not None and int(rows_cfg) <= 0:
        raise ValueError("--rows must be a positive integer when provided.")

    run_name = args.run_name or args.method

    data_dir = _resolve_path(args.data_outdir, cfg, "paths.data_dir", "data")
    output_root = _resolve_path(args.outdir, cfg, "paths.output_dir", "outputs")
    reports_dir = _resolve_path(args.report_dir, cfg, "paths.report_dir", "reports")

    run_dir = output_root / run_name
    plots_dir = run_dir / "plots"

    # Important:
    # Do not create output/report folders before --validate-only exits.
    # Validation mode should check config/data/schema only and should not leave
    # empty folders like outputs/copula/plots behind.
    df = load_or_generate(
        args.data,
        seed=seed,
        generated_output_path=data_dir / "real_data.csv",
    )
    validate_dataframe(df)

    numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=cat_thr)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable columns were detected in the input data.")

    n_rows = int(rows_cfg) if rows_cfg is not None else len(df)

    if args.validate_only:
        print("Validation successful.")
        print(f"Rows: {len(df)}")
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
        return

    # Create output folders only when a generation/evaluation run is actually happening.
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "copula":
        df_syn = generate_copula(
            df,
            numeric_cols,
            categorical_cols,
            n_rows=n_rows,
            seed=seed,
        )
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

    validate_dataframe(df_syn)

    syn_path = data_dir / f"synthetic_data_{run_name}.csv"
    df_syn.to_csv(syn_path, index=False)

    dist_scores = plot_distribution_overlap(
        df,
        df_syn,
        bins=bins,
        out_path=plots_dir / "distribution_overlap.png",
        numeric_cols=numeric_cols,
    )

    pca_info = plot_pca(
        df,
        df_syn,
        out_path=plots_dir / "pca_projection.png",
        n_components=pca_components,
        numeric_cols=numeric_cols,
    )

    corr_info = plot_correlation_heatmap(
        df,
        df_syn,
        out_path=plots_dir / "correlation_heatmap.png",
        numeric_cols=numeric_cols,
    )

    pairplot_enabled = bool(get_nested(cfg, "plots.pairplot", True)) and not args.skip_pairplot

    if pairplot_enabled:
        pairplot_compare(
            df,
            df_syn,
            out_path=plots_dir / "pairplot_comparison.png",
            sample=pairplot_sample,
            numeric_cols=numeric_cols,
        )

    categorical_info = categorical_distribution_similarity(df, df_syn, categorical_cols)
    numeric_summary_info = numeric_summary_differences(df, df_syn, numeric_cols)
    boundary_info = boundary_violation_rates(df, df_syn, numeric_cols, categorical_cols)

    privacy_info = privacy_nearest_neighbor_metrics(
        df,
        df_syn,
        numeric_cols,
        categorical_cols,
        max_rows=int(get_nested(cfg, "evaluation.privacy_max_rows", 500)),
        seed=seed,
    )

    ml_target = get_nested(cfg, "evaluation.ml_utility.target", None)

    if ml_target:
        ml_utility_info = ml_utility_metrics(
            df,
            df_syn,
            target_col=str(ml_target),
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            seed=seed,
            test_size=float(get_nested(cfg, "evaluation.ml_utility.test_size", 0.25)),
        )
    else:
        ml_utility_info = {
            "ml_utility_available": False,
            "ml_utility_reason": "No target configured at evaluation.ml_utility.target.",
        }

    distribution_values = [value for value in dist_scores.values() if value is not None]

    metrics = {
        "rows_real": int(len(df)),
        "rows_synthetic": int(len(df_syn)),
        "method": args.method,
        "seed": seed,
        "schema": {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        "paths": {
            "synthetic_data": str(syn_path),
            "run_dir": str(run_dir),
            "plots_dir": str(plots_dir),
            "report": str(reports_dir / f"{run_name}_report.html"),
        },
        "pairplot_enabled": pairplot_enabled,
        "distribution_overlap_mean": (
            float(np.mean(distribution_values)) if distribution_values else None
        ),
        "distribution_overlap_per_feature": dist_scores,
        "pca_explained_variance": (pca_info or {}).get("explained_variance"),
        **corr_info,
        **categorical_info,
        **numeric_summary_info,
        **boundary_info,
        **privacy_info,
        **ml_utility_info,
    }

    save_json(run_dir / "metrics.json", metrics)

    summary_keys = [
        "method",
        "rows_real",
        "rows_synthetic",
        "distribution_overlap_mean",
        "correlation_diff_mean",
        "categorical_similarity_mean",
        "numeric_summary_diff_mean",
        "boundary_violation_rate_mean",
        "exact_duplicate_rate",
        "nearest_neighbor_distance_p05",
        "ml_utility_available",
        "utility_ratio",
    ]

    summary = {key: metrics.get(key) for key in summary_keys if key in metrics}
    pd.DataFrame([summary]).to_csv(run_dir / "quality_summary.csv", index=False)

    report_path = reports_dir / f"{run_name}_report.html"

    write_report(
        method=args.method,
        rows=n_rows,
        seed=seed,
        metrics=metrics,
        report_path=report_path,
        run_name=run_name,
    )

    print(f"Saved synthetic CSV: {syn_path}")
    print(f"Saved metrics:       {run_dir / 'metrics.json'}")
    print(f"Saved plots in:      {plots_dir}")
    print(f"Saved report:        {report_path}")


if __name__ == "__main__":
    main()
