"""Evaluation metrics for synthetic-data quality.

This package was split out of a single module; the public API is unchanged and
is re-exported here so that ``from ...evaluation.metrics import name`` continues
to work exactly as before.
"""

from __future__ import annotations

from ._common import _as_clean_numeric, _make_one_hot_encoder
from .distribution import (
    boundary_violation_rates,
    categorical_distribution_similarity,
    correlation_diff_mean,
    distribution_overlap_scores,
    js_distance,
    numeric_summary_differences,
)
from .privacy import privacy_nearest_neighbor_metrics
from .utility import ml_utility_metrics

__all__ = [
    "_as_clean_numeric",
    "_make_one_hot_encoder",
    "boundary_violation_rates",
    "categorical_distribution_similarity",
    "correlation_diff_mean",
    "distribution_overlap_scores",
    "js_distance",
    "ml_utility_metrics",
    "numeric_summary_differences",
    "privacy_nearest_neighbor_metrics",
]
