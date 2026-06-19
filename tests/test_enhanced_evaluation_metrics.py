from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from synthetic_data_artist.evaluation.metrics import (
    boundary_violation_rates,
    categorical_distribution_similarity,
    distribution_overlap_scores,
    ml_utility_metrics,
    numeric_summary_differences,
    privacy_nearest_neighbor_metrics,
)


class EnhancedEvaluationMetricsTests(unittest.TestCase):
    def _real_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 80
        return pd.DataFrame(
            {
                "age": rng.normal(40, 8, n).round(2),
                "income": rng.normal(75_000, 10_000, n).round(2),
                "segment": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
                "channel": rng.choice(["web", "store"], n),
                "target": rng.choice(["yes", "no"], n),
            }
        )

    def test_categorical_distribution_similarity_is_bounded(self) -> None:
        real = self._real_df()
        synthetic = real.sample(frac=1.0, random_state=1).reset_index(drop=True)

        result = categorical_distribution_similarity(real, synthetic, ["segment", "channel"])

        self.assertIn("categorical_similarity_mean", result)
        self.assertGreaterEqual(result["categorical_similarity_mean"], 0.0)
        self.assertLessEqual(result["categorical_similarity_mean"], 1.0)
        self.assertEqual(set(result["categorical_similarity_per_feature"]), {"segment", "channel"})

    def test_numeric_summary_and_distribution_metrics_return_expected_keys(self) -> None:
        real = self._real_df()
        synthetic = real.copy()
        synthetic["income"] = synthetic["income"] * 1.05

        overlap = distribution_overlap_scores(real, synthetic, numeric_cols=["age", "income"])
        summary = numeric_summary_differences(real, synthetic, ["age", "income"])

        self.assertEqual(set(overlap), {"age", "income"})
        self.assertIn("numeric_summary_diff_mean", summary)
        self.assertIn("income", summary["numeric_summary_diff_per_feature"])

    def test_boundary_violation_rates_detect_invalid_values(self) -> None:
        real = self._real_df()
        synthetic = real.copy()
        synthetic.loc[0, "age"] = 999
        synthetic.loc[1, "segment"] = "INVALID"

        result = boundary_violation_rates(
            real,
            synthetic,
            numeric_cols=["age", "income"],
            categorical_cols=["segment", "channel"],
        )

        self.assertGreater(result["numeric_boundary_violation_rate"]["age"], 0.0)
        self.assertGreater(result["categorical_invalid_rate"]["segment"], 0.0)
        self.assertGreater(result["boundary_violation_rate_mean"], 0.0)

    def test_privacy_metrics_return_proxy_values(self) -> None:
        real = self._real_df()
        synthetic = real.head(20).copy()

        result = privacy_nearest_neighbor_metrics(
            real,
            synthetic,
            numeric_cols=["age", "income"],
            categorical_cols=["segment", "channel"],
            max_rows=40,
            seed=42,
        )

        self.assertIn("privacy_note", result)
        self.assertGreaterEqual(result["exact_duplicate_rate"], 0.0)
        self.assertLessEqual(result["exact_duplicate_rate"], 1.0)
        self.assertGreaterEqual(result["nearest_neighbor_distance_mean"], 0.0)

    def test_ml_utility_handles_missing_target(self) -> None:
        real = self._real_df()
        synthetic = real.copy()

        result = ml_utility_metrics(
            real,
            synthetic,
            target_col="missing",
            numeric_cols=["age", "income"],
            categorical_cols=["segment", "channel", "target"],
        )

        self.assertFalse(result["ml_utility_available"])

    def test_ml_utility_runs_for_classification_target(self) -> None:
        real = self._real_df()
        synthetic = real.sample(frac=1.0, random_state=2).reset_index(drop=True)

        result = ml_utility_metrics(
            real,
            synthetic,
            target_col="target",
            numeric_cols=["age", "income"],
            categorical_cols=["segment", "channel", "target"],
            seed=42,
            test_size=0.3,
        )

        self.assertTrue(result["ml_utility_available"])
        self.assertEqual(result["ml_utility_metric"], "accuracy")
        self.assertIn("train_real_test_real", result)
        self.assertIn("train_synthetic_test_real", result)


if __name__ == "__main__":
    unittest.main()
