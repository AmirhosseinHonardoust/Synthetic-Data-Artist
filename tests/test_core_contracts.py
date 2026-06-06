from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_MAIN = ROOT / "src" / "main.py"

spec = importlib.util.spec_from_file_location("synthetic_artist_main", SRC_MAIN)
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["synthetic_artist_main"] = app
spec.loader.exec_module(app)


class CoreContractsTests(unittest.TestCase):
    def _demo_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 80
        return pd.DataFrame(
            {
                "age": rng.normal(40, 10, n).round(2),
                "income": rng.normal(75_000, 12_000, n).round(2),
                "segment": rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20]),
                "visits": rng.integers(1, 6, size=n),
            }
        )

    def test_detect_schema_reclassifies_low_cardinality_integer_columns(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = app.detect_schema(df, categorical_threshold=10)

        self.assertIn("age", numeric_cols)
        self.assertIn("income", numeric_cols)
        self.assertIn("segment", categorical_cols)
        self.assertIn("visits", categorical_cols)
        self.assertEqual(set(numeric_cols + categorical_cols), set(df.columns))

    def test_copula_output_preserves_contract(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = app.detect_schema(df, categorical_threshold=10)

        synthetic = app.generate_copula(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_rows=25,
            seed=123,
        )

        self.assertEqual(list(synthetic.columns), list(df.columns))
        self.assertEqual(len(synthetic), 25)
        for col in categorical_cols:
            self.assertTrue(set(synthetic[col].unique()).issubset(set(df[col].unique())))
        for col in numeric_cols:
            self.assertTrue(np.isfinite(synthetic[col].to_numpy(dtype=float)).all())

    def test_metric_and_report_helpers_write_expected_files(self) -> None:
        df = self._demo_df()
        numeric_cols, categorical_cols = app.detect_schema(df, categorical_threshold=10)
        synthetic = app.generate_copula(df, numeric_cols, categorical_cols, n_rows=30, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dist = app.plot_distribution_overlap(
                df,
                synthetic,
                bins=8,
                out_path=tmp / "plots" / "distribution_overlap.png",
            )
            pca = app.plot_pca(df, synthetic, out_path=tmp / "plots" / "pca_projection.png")
            corr = app.plot_correlation_heatmap(df, synthetic, out_path=tmp / "plots" / "correlation_heatmap.png")
            metrics = {"distribution_overlap_mean": float(np.mean(list(dist.values()))), **corr, **(pca or {})}
            app.write_report("copula", len(synthetic), 42, metrics, tmp / "report.html", "ci_test")

            self.assertTrue((tmp / "plots" / "distribution_overlap.png").exists())
            self.assertTrue((tmp / "plots" / "pca_projection.png").exists())
            self.assertTrue((tmp / "plots" / "correlation_heatmap.png").exists())
            self.assertTrue((tmp / "report.html").exists())

    def test_vae_output_preserves_contract_on_small_data(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"PyTorch is not installed: {exc}")

        df = self._demo_df().head(32)
        numeric_cols, categorical_cols = app.detect_schema(df, categorical_threshold=10)

        synthetic = app.train_and_generate_vae(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_rows=12,
            seed=42,
            epochs=1,
            batch=16,
            latent=2,
        )

        self.assertEqual(list(synthetic.columns), list(df.columns))
        self.assertEqual(len(synthetic), 12)
        for col in categorical_cols:
            self.assertTrue(set(synthetic[col].unique()).issubset(set(df[col].unique())))


if __name__ == "__main__":
    unittest.main()
