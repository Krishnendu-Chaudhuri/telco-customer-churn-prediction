"""Tests for champion/challenger model registry."""

from __future__ import annotations

import pytest

from src.models.db import get_connection, init_schema
from src.models.registry import ModelRegistry


def _result(
    model_name: str,
    roc_auc: float,
    recall: float,
    *,
    trained_at: str = "2026-06-10T12:00:00+00:00",
) -> dict:
    return {
        "model_name": model_name,
        "metrics": {
            "roc_auc": roc_auc,
            "recall": recall,
            "accuracy": 0.8,
            "precision": 0.75,
            "f1": 0.72,
        },
        "trained_at": trained_at,
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


@pytest.fixture
def registry(tmp_path, registry_config: dict) -> ModelRegistry:
    registry_path = tmp_path / "registry.db"
    return ModelRegistry(config=registry_config, registry_path=registry_path)


def test_initial_run_picks_higher_metric(registry: ModelRegistry) -> None:
    lr = _result("logistic_regression", roc_auc=0.839, recall=0.75)
    lgbm = _result("lightgbm", roc_auc=0.835, recall=0.80)

    decision = registry.evaluate_and_decide(lr, lgbm)

    assert decision["action"] == "initial"
    assert registry.get_current_champion_name() == "logistic_regression"
    assert registry.get_current_challenger_name() == "lightgbm"
    assert registry.history[-1]["action"] == "initial"
    assert registry.history[-1]["new_champion"] == "logistic_regression"


def test_promote_when_threshold_met(registry: ModelRegistry) -> None:
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75),
        _result("lightgbm", roc_auc=0.835, recall=0.80),
    )

    decision = registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.830, recall=0.74),
        _result("lightgbm", roc_auc=0.840, recall=0.72),
    )

    assert decision["action"] == "promote"
    assert registry.get_current_champion_name() == "lightgbm"
    assert registry.get_current_challenger_name() == "logistic_regression"
    assert registry.history[-1]["delta"] == pytest.approx(0.01)
    assert registry.history[-1]["action"] == "promote"


def test_retain_when_recall_below_floor(registry: ModelRegistry) -> None:
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75),
        _result("lightgbm", roc_auc=0.835, recall=0.80),
    )

    decision = registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.830, recall=0.75),
        _result("lightgbm", roc_auc=0.845, recall=0.65),
    )

    assert decision["action"] == "retain"
    assert registry.get_current_champion_name() == "logistic_regression"
    assert registry.history[-1]["action"] == "retain"


def test_retain_when_delta_below_threshold(registry: ModelRegistry) -> None:
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75),
        _result("lightgbm", roc_auc=0.835, recall=0.80),
    )

    decision = registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75),
        _result("lightgbm", roc_auc=0.841, recall=0.72),
    )

    assert decision["action"] == "retain"
    assert registry.get_current_champion_name() == "logistic_regression"
    assert registry.history[-1]["delta"] == pytest.approx(0.002)


def test_rollback_swaps_roles(registry: ModelRegistry) -> None:
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75),
        _result("lightgbm", roc_auc=0.835, recall=0.80),
    )

    decision = registry.rollback()

    assert decision["action"] == "manual_rollback"
    assert registry.get_current_champion_name() == "lightgbm"
    assert registry.get_current_challenger_name() == "logistic_regression"
    assert registry.history[-1]["action"] == "manual_rollback"


def test_save_load_preserves_history(
    tmp_path,
    registry_config: dict,
) -> None:
    registry_path = tmp_path / "registry.db"
    registry = ModelRegistry(config=registry_config, registry_path=registry_path)

    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75, trained_at="t1"),
        _result("lightgbm", roc_auc=0.835, recall=0.80, trained_at="t1"),
    )
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.839, recall=0.75, trained_at="t2"),
        _result("lightgbm", roc_auc=0.841, recall=0.72, trained_at="t2"),
    )
    registry.evaluate_and_decide(
        _result("logistic_regression", roc_auc=0.830, recall=0.74, trained_at="t3"),
        _result("lightgbm", roc_auc=0.840, recall=0.72, trained_at="t3"),
    )

    assert len(registry.history) == 3

    reloaded = ModelRegistry.load(config=registry_config, registry_path=registry_path)
    assert reloaded.get_current_champion_name() == registry.get_current_champion_name()
    assert reloaded.get_current_challenger_name() == registry.get_current_challenger_name()
    assert len(reloaded.history) == 3
    assert reloaded.promotion_threshold == 0.005

    conn = get_connection(registry_path)
    try:
        init_schema(conn)
        history_count = conn.execute("SELECT COUNT(*) FROM promotion_history").fetchone()[0]
        role_count = conn.execute("SELECT COUNT(*) FROM model_roles").fetchone()[0]
    finally:
        conn.close()
    assert history_count == 3
    assert role_count == 2
