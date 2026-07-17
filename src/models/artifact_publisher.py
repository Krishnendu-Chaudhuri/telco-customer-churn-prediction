"""Atomic artifact publishing for training outputs."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class ArtifactPublisher:
    """Write training artifacts to staging, then atomically publish to live paths."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def create_staging_dir(self) -> Path:
        """Create a unique staging directory under models/."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        staging_dir = self.paths.models_dir / f".staging_{timestamp}"
        (staging_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        logger.info("Created staging directory: %s", staging_dir)
        return staging_dir

    def staging_paths(self, staging_dir: Path) -> dict[str, Path]:
        """Resolve staging file paths mirroring live artifact layout."""
        return {
            "best_model": staging_dir / "best_model.pkl",
            "scaler": staging_dir / "scaler.pkl",
            "encoder": staging_dir / "encoder.pkl",
            "feature_pipeline": staging_dir / "feature_pipeline.pkl",
            "model_comparison": staging_dir / "model_comparison.csv",
            "model_metadata": staging_dir / "model_metadata.json",
            "registry_db": staging_dir / "registry.db",
            "registry_dir": staging_dir / "registry",
            "kmeans_model": staging_dir / "kmeans_model.pkl",
            "segment_mapping": staging_dir / "segment_mapping.json",
            "segment_assignments": staging_dir / "segment_assignments.csv",
            "processed_data": staging_dir / "processed_customers.parquet",
            "evaluation_dir": staging_dir / "evaluation",
        }

    def publish(self, staging_dir: Path) -> None:
        """Atomically move staged artifacts into live directories."""
        staging = self.staging_paths(staging_dir)
        file_moves = [
            (staging["best_model"], self.paths.best_model),
            (staging["scaler"], self.paths.scaler),
            (staging["encoder"], self.paths.encoder),
            (staging["feature_pipeline"], self.paths.feature_pipeline),
            (staging["model_comparison"], self.paths.model_comparison),
            (staging["model_metadata"], self.paths.model_metadata),
            (staging["registry_db"], self.paths.registry_db),
            (staging["kmeans_model"], self.paths.kmeans_model),
            (staging["segment_mapping"], self.paths.segment_mapping),
            (staging["segment_assignments"], self.paths.segment_assignments),
            (staging["processed_data"], self.paths.processed_data),
        ]

        self.paths.ensure_dirs()
        for src, dst in file_moves:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)

        eval_staging = staging["evaluation_dir"]
        if eval_staging.exists():
            self.paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
            for item in eval_staging.iterdir():
                os.replace(item, self.paths.evaluation_dir / item.name)

        registry_staging = staging["registry_dir"]
        if registry_staging.exists():
            self.paths.registry_dir.mkdir(parents=True, exist_ok=True)
            for item in registry_staging.iterdir():
                if item.is_file():
                    os.replace(item, self.paths.registry_dir / item.name)

        self.cleanup(staging_dir)
        logger.info("Published artifacts from staging to live paths")

    def cleanup(self, staging_dir: Path) -> None:
        """Remove a staging directory."""
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
            logger.info("Cleaned up staging directory: %s", staging_dir)
