"""Tests for atomic training artifact publishing."""

from __future__ import annotations

import hashlib

import pytest

from src.models.train_model import train
from src.utils.paths import ProjectPaths


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.slow
def test_failed_training_leaves_live_artifacts_unchanged(monkeypatch):
    paths = ProjectPaths()
    if not paths.best_model.exists():
        pytest.skip("No existing model artifacts to protect")

    original_hash = _file_sha256(paths.best_model)
    registry_hash = (
        _file_sha256(paths.registry_db) if paths.registry_db.exists() else None
    )

    def failing_tune(*_args, **_kwargs):
        raise RuntimeError("Simulated training failure")

    monkeypatch.setattr("src.models.train_model.tune_models", failing_tune)

    with pytest.raises(RuntimeError, match="Simulated training failure"):
        train()

    assert paths.best_model.exists()
    assert _file_sha256(paths.best_model) == original_hash
    if registry_hash is not None:
        assert paths.registry_db.exists()
        assert _file_sha256(paths.registry_db) == registry_hash

    staging_dirs = list(paths.models_dir.glob(".staging_*"))
    assert staging_dirs == []
