"""Project path resolution utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_MARKERS = ("app", "src", "configs")


def _resolve_project_root(anchor: Path) -> Path:
    """Walk up from anchor until the repository root markers are found."""
    start = anchor.resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in _REPO_MARKERS):
            return candidate

    raise RuntimeError(
        f"Could not resolve project root from {anchor}. "
        f"Expected a parent directory containing {_REPO_MARKERS}."
    )


def get_project_root() -> Path:
    """Return the project root directory."""
    return _resolve_project_root(Path(__file__))


def ensure_project_imports(anchor: Path | None = None) -> Path:
    """Pin this repository on sys.path and clear stale cross-project imports."""
    root = _resolve_project_root(anchor or Path(__file__))
    root_str = str(root)

    sys.path[:] = [entry for entry in sys.path if entry != root_str]
    sys.path.insert(0, root_str)
    os.chdir(root)

    for name in list(sys.modules):
        if name in {"app", "src"} or name.startswith(("app.", "src.")):
            module = sys.modules.get(name)
            if module is None:
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                Path(module_file).resolve().relative_to(root)
            except ValueError:
                del sys.modules[name]

    return root


class ProjectPaths:
    """Resolve configured project paths relative to project root."""

    def __init__(self, config: dict | None = None) -> None:
        from src.utils.config import get_config

        self.config = config or get_config()
        self.root = get_project_root()
        paths = self.config["paths"]

        self.raw_data = self.root / paths["raw_data"]
        self.processed_data = self.root / paths["processed_data"]
        self.models_dir = self.root / paths["models_dir"]
        self.evaluation_dir = self.root / paths["evaluation_dir"]
        self.logs_dir = self.root / paths["logs_dir"]

        self.best_model = self.models_dir / "best_model.pkl"
        self.scaler = self.models_dir / "scaler.pkl"
        self.encoder = self.models_dir / "encoder.pkl"
        self.feature_pipeline = self.models_dir / "feature_pipeline.pkl"
        self.kmeans_model = self.models_dir / "kmeans_model.pkl"
        self.model_metadata = self.models_dir / "model_metadata.json"
        self.model_comparison = self.models_dir / "model_comparison.csv"
        self.segment_mapping = self.models_dir / "segment_mapping.json"
        self.segment_assignments = self.models_dir / "segment_assignments.csv"
        self.registry_dir = self.models_dir / "registry"
        self.registry_json = self.models_dir / "registry.json"
        self.registry_db = self.models_dir / "registry.db"
        self.training_lock = self.models_dir / ".training.lock"

    def model_artifact(self, model_name: str) -> Path:
        """Return path to a model-name-based artifact under models/registry/."""
        return self.registry_dir / f"{model_name}.pkl"

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        for path in (
            self.models_dir,
            self.registry_dir,
            self.evaluation_dir,
            self.logs_dir,
            self.processed_data.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
