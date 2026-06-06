from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class ConfigValidationError(ValueError):
    """Raised when config values are missing, invalid, or inconsistent."""


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return an empty dict when the file is empty."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Save a dictionary as formatted JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_nested(config: dict[str, Any], key: str, default: Any) -> Any:
    """Read a nested config value using dot notation, e.g. 'vae.epochs'."""
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def validate_config(config: dict[str, Any]) -> None:
    """Validate common config values and raise a clear error when invalid.

    The config remains intentionally lightweight, but invalid settings such as
    negative row counts or impossible train/test splits should fail early.
    """
    errors: list[str] = []

    if "rows" in config and config["rows"] is not None and not _is_positive_int(config["rows"]):
        errors.append("rows must be a positive integer or null")

    if "categorical_threshold" in config and not _is_positive_int(config["categorical_threshold"]):
        errors.append("categorical_threshold must be a positive integer")

    if "seed" in config and not isinstance(config["seed"], int):
        errors.append("seed must be an integer")

    if "pca_components" in config and not _is_positive_int(config["pca_components"]):
        errors.append("pca_components must be a positive integer")

    if "hist_bins" in config and not _is_positive_int(config["hist_bins"]):
        errors.append("hist_bins must be a positive integer")

    if "pairplot_sample" in config and not _is_positive_int(config["pairplot_sample"]):
        errors.append("pairplot_sample must be a positive integer")

    vae = config.get("vae", {}) or {}
    if not isinstance(vae, dict):
        errors.append("vae must be a mapping")
    else:
        for key in ["epochs", "batch_size", "latent_dim", "hidden_dim"]:
            if key in vae and not _is_positive_int(vae[key]):
                errors.append(f"vae.{key} must be a positive integer")

        for key in ["learning_rate", "kl_weight"]:
            if key in vae and not _is_positive_number(vae[key]):
                errors.append(f"vae.{key} must be a positive number")

    paths = config.get("paths", {}) or {}
    if not isinstance(paths, dict):
        errors.append("paths must be a mapping")
    else:
        for key in ["data_dir", "output_dir", "report_dir"]:
            if key in paths and not isinstance(paths[key], str):
                errors.append(f"paths.{key} must be a string")

    plots = config.get("plots", {}) or {}
    if not isinstance(plots, dict):
        errors.append("plots must be a mapping")
    elif "pairplot" in plots and not isinstance(plots["pairplot"], bool):
        errors.append("plots.pairplot must be true or false")

    evaluation = config.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be a mapping")
    else:
        privacy_max_rows = evaluation.get("privacy_max_rows")
        if privacy_max_rows is not None and not _is_positive_int(privacy_max_rows):
            errors.append("evaluation.privacy_max_rows must be a positive integer")

        ml_utility = evaluation.get("ml_utility", {}) or {}
        if not isinstance(ml_utility, dict):
            errors.append("evaluation.ml_utility must be a mapping")
        else:
            test_size = ml_utility.get("test_size")
            if test_size is not None and not (
                isinstance(test_size, (int, float)) and not isinstance(test_size, bool) and 0 < float(test_size) < 1
            ):
                errors.append("evaluation.ml_utility.test_size must be between 0 and 1")

            target = ml_utility.get("target")
            if target is not None and not isinstance(target, str):
                errors.append("evaluation.ml_utility.target must be a string or null")

    if errors:
        raise ConfigValidationError("Invalid config: " + "; ".join(errors))
