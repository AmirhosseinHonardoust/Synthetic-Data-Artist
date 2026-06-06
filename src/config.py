from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


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
