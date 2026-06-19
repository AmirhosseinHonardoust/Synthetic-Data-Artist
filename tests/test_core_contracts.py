from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from synthetic_data_artist.evaluation.plots import (
    plot_correlation_heatmap,
    plot_distribution_overlap,
    plot_pca,
)
from synthetic_data_artist.models.copula import generate_copula
from synthetic_data_artist.models.vae import train_and_generate_vae
from synthetic_data_artist.reporting.html_report import write_report
from synthetic_data_artist.schema import detect_schema


class CoreContractsTests(unittest.TestCase):
    def _demo_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 80

        return pd.DataFrame(
            {
                "age": rng.normal(40, 10, n).round(2),
                "income": rng.normal(75_000, 12_000, n).round(2),
                "score": rng.normal(0, 1, n).round(4),
                "segment": rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20]),
                "channel": rng.choice(["web", "store", "mobile"], size=n),
                "visits": rng.integers(1, 6, size=n),
            }
        )

    def test_detect_schema_reclassifies_low_cardinality_integer_columns(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=10)

        self.assertIn("age", numeric_cols)
        self.assertIn("income", numeric_cols)
        self.assertIn("score", numeric_cols)
        self.assertIn("segment", categorical_cols)
        self.assertIn("channel", categorical_cols)
        self.assertIn("visits", categorical_cols)
        self.assertEqual(set(numeric_cols + categorical_cols), set(df.columns))

    def test_copula_output_preserves_contract(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=10)

        synthetic = generate_copula(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_rows=25,
            seed=123,
        )

        self.assertEqual(list(synthetic.columns), list(df.columns))
        self.assertEqual(len(synthetic), 25)

        for col in categorical_cols:
            self.assertTrue(
                set(synthetic[col].unique()).issubset(set(df[col].unique())),
                msg=f"Synthetic categorical values for {col!r} should come from the real data.",
            )

        for col in numeric_cols:
            self.assertTrue(
                np.isfinite(synthetic[col].to_numpy(dtype=float)).all(),
                msg=f"Synthetic numeric values for {col!r} should be finite.",
            )

    def test_metric_and_report_helpers_write_expected_files(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=10)
        synthetic = generate_copula(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_rows=30,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plots_dir = tmp / "plots"

            dist = plot_distribution_overlap(
                df,
                synthetic,
                bins=8,
                out_path=plots_dir / "distribution_overlap.png",
            )
            pca = plot_pca(
                df,
                synthetic,
                out_path=plots_dir / "pca_projection.png",
            )
            corr = plot_correlation_heatmap(
                df,
                synthetic,
                out_path=plots_dir / "correlation_heatmap.png",
            )

            metrics = {
                "distribution_overlap_mean": float(np.mean(list(dist.values()))),
                **(pca or {}),
                **corr,
            }

            write_report(
                method="copula",
                rows=len(synthetic),
                seed=42,
                metrics=metrics,
                report_path=tmp / "report.html",
                run_name="ci_test",
            )

            self.assertTrue((plots_dir / "distribution_overlap.png").exists())
            self.assertTrue((plots_dir / "pca_projection.png").exists())
            self.assertTrue((plots_dir / "correlation_heatmap.png").exists())
            self.assertTrue((tmp / "report.html").exists())

    def test_vae_output_preserves_contract_on_small_data(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"PyTorch is not installed: {exc}")

        df = self._demo_df().head(32)
        numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=10)

        synthetic = train_and_generate_vae(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_rows=12,
            seed=42,
            epochs=1,
            batch=16,
            latent=2,
            hidden=16,
            learning_rate=0.001,
            kl_weight=0.001,
        )

        self.assertEqual(list(synthetic.columns), list(df.columns))
        self.assertEqual(len(synthetic), 12)

        for col in categorical_cols:
            self.assertTrue(
                set(synthetic[col].unique()).issubset(set(df[col].unique())),
                msg=f"Synthetic categorical values for {col!r} should come from the real data.",
            )

        for col in numeric_cols:
            self.assertTrue(
                np.isfinite(synthetic[col].to_numpy(dtype=float)).all(),
                msg=f"Synthetic numeric values for {col!r} should be finite.",
            )


if __name__ == "__main__":
    unittest.main()
