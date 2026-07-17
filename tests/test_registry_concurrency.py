"""Concurrent write tests for SQLite-backed model registry."""

from __future__ import annotations

import threading

import pytest

from src.models.db import get_connection, init_schema
from src.models.registry import ModelRegistry


def _result(model_name: str, roc_auc: float, recall: float) -> dict:
    return {
        "model_name": model_name,
        "metrics": {
            "roc_auc": roc_auc,
            "recall": recall,
            "accuracy": 0.8,
            "precision": 0.75,
            "f1": 0.72,
        },
        "trained_at": "2026-06-10T12:00:00+00:00",
    }


@pytest.fixture
def registry_config() -> dict:
    return {
        "champion_challenger": {
            "primary_metric": "roc_auc",
            "promotion_threshold": 0.005,
            "min_recall_floor": 0.70,
        },
        "paths": {
            "raw_data": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "processed_data": "data/processed/processed_customers.parquet",
            "models_dir": "models",
            "evaluation_dir": "models/evaluation",
            "logs_dir": "logs",
        },
    }


def test_concurrent_evaluate_and_decide_is_consistent(
    tmp_path,
    registry_config: dict,
) -> None:
    registry_path = tmp_path / "registry.db"
    errors: list[Exception] = []

    def worker(offset: float) -> None:
        try:
            registry = ModelRegistry.load(config=registry_config, registry_path=registry_path)
            if registry.get_current_champion_name() is None:
                registry.evaluate_and_decide(
                    _result("logistic_regression", 0.839 + offset, 0.75),
                    _result("lightgbm", 0.835 + offset, 0.80),
                )
            else:
                registry.evaluate_and_decide(
                    _result("logistic_regression", 0.830 + offset, 0.74),
                    _result("lightgbm", 0.840 + offset, 0.72),
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i * 0.001,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    final = ModelRegistry.load(config=registry_config, registry_path=registry_path)
    assert final.get_current_champion_name() is not None
    assert final.get_current_challenger_name() is not None
    assert len(final.history) >= 1

    conn = get_connection(registry_path)
    try:
        init_schema(conn)
        history_count = conn.execute("SELECT COUNT(*) FROM promotion_history").fetchone()[0]
        role_count = conn.execute("SELECT COUNT(*) FROM model_roles").fetchone()[0]
    finally:
        conn.close()

    assert history_count == len(final.history)
    assert role_count == 2
