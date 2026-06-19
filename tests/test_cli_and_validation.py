from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synthetic_data_artist.config import ConfigValidationError, validate_config
from synthetic_data_artist.data import DataValidationError, load_or_generate, validate_dataframe
from synthetic_data_artist.main import parse_args


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config_passes(self) -> None:
        validate_config(
            {
                "rows": 100,
                "categorical_threshold": 10,
                "seed": 42,
                "pca_components": 2,
                "hist_bins": 20,
                "pairplot_sample": 50,
                "paths": {
                    "data_dir": "data",
                    "output_dir": "outputs",
                    "report_dir": "reports",
                },
                "plots": {"pairplot": False},
                "vae": {
                    "epochs": 1,
                    "batch_size": 16,
                    "latent_dim": 2,
                    "hidden_dim": 16,
                    "learning_rate": 0.001,
                    "kl_weight": 0.001,
                },
                "evaluation": {
                    "privacy_max_rows": 100,
                    "ml_utility": {"target": None, "test_size": 0.25},
                },
            }
        )

    def test_invalid_config_raises_clear_error(self) -> None:
        with self.assertRaises(ConfigValidationError) as ctx:
            validate_config(
                {
                    "rows": -1,
                    "categorical_threshold": 0,
                    "vae": {"epochs": 0, "learning_rate": -0.1},
                    "evaluation": {"ml_utility": {"test_size": 1.5}},
                }
            )

        message = str(ctx.exception)
        self.assertIn("rows", message)
        self.assertIn("categorical_threshold", message)
        self.assertIn("vae.epochs", message)
        self.assertIn("evaluation.ml_utility.test_size", message)


class DataValidationTests(unittest.TestCase):
    def test_valid_dataframe_passes(self) -> None:
        df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "segment": ["A", "B", "A"],
            }
        )
        validate_dataframe(df)

    def test_empty_dataframe_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_dataframe(pd.DataFrame())

    def test_all_null_column_raises(self) -> None:
        df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "bad": [None, None, None],
            }
        )
        with self.assertRaises(DataValidationError) as ctx:
            validate_dataframe(df)
        self.assertIn("entirely missing", str(ctx.exception))

    def test_duplicate_column_raises(self) -> None:
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        with self.assertRaises(DataValidationError) as ctx:
            validate_dataframe(df)
        self.assertIn("duplicate", str(ctx.exception))

    def test_load_or_generate_can_write_demo_data_to_custom_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing_csv = tmp / "missing.csv"
            generated_csv = tmp / "generated" / "real_data.csv"

            df = load_or_generate(
                missing_csv,
                seed=42,
                generated_output_path=generated_csv,
            )

            self.assertTrue(generated_csv.exists())
            self.assertGreater(len(df), 0)
            self.assertIn("target", df.columns)


class CliParsingTests(unittest.TestCase):
    def test_parse_args_supports_output_and_fast_mode_options(self) -> None:
        args = parse_args(
            [
                "--method",
                "copula",
                "--run_name",
                "demo",
                "--rows",
                "25",
                "--outdir",
                "custom_outputs",
                "--data-outdir",
                "custom_data",
                "--report-dir",
                "custom_reports",
                "--skip-pairplot",
                "--validate-only",
            ]
        )

        self.assertEqual(args.method, "copula")
        self.assertEqual(args.run_name, "demo")
        self.assertEqual(args.rows, 25)
        self.assertEqual(args.outdir, "custom_outputs")
        self.assertEqual(args.data_outdir, "custom_data")
        self.assertEqual(args.report_dir, "custom_reports")
        self.assertTrue(args.skip_pairplot)
        self.assertTrue(args.validate_only)


if __name__ == "__main__":
    unittest.main()
