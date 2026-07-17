"""Configuration loader for the churn retention engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import get_project_root


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return cached configuration."""
    return load_config()
