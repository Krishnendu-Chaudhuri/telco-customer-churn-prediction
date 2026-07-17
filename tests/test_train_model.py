"""Unit tests for train_model.train() with mocked tuning pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelRegistry
from src.models.train_model import train
from src.utils.config import get_config


class _MockSearch:
    def __init__(self, model: LogisticRegression) -> None:
        self.best_estimator_ = model
        self.best_params_ = {"C": 1.0}
        self.best_score_ = 0.85


def _mock_tune_models(x_train, y_train, config=None):
    lr_model = LogisticRegression(max_iter=200).fit(x_train, y_train)
    lgbm_model = LogisticRegression(max_iter=200).fit(x_train, y_train)
    return {
        "logistic_regression": _MockSearch(lr_model),
        "lightgbm": _MockSearch(lgbm_model),
    }


def _training_config(tmp_path) -> dict:
    config = get_config()
    config = {
        **config,
        "paths": {
            **config["paths"],
            "models_dir": str(tmp_path / "models"),
            "evaluation_dir": str(tmp_path / "models" / "evaluation"),
            "logs_dir": str(tmp_path / "logs"),
            "processed_data": str(tmp_path / "data" / "processed_customers.parquet"),
        },
    }
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def train_mocks(monkeypatch, sample_df):
    monkeypatch.setattr("src.models.train_model.tune_models", _mock_tune_models)
    monkeypatch.setattr("src.models.train_model.load_raw_data", lambda: sample_df)
    monkeypatch.setattr("src.models.train_model._log_to_mlflow", lambda *args, **kwargs: None)

    mock_explainer = MagicMock()
    mock_explainer.save_global_summary = MagicMock()
    monkeypatch.setattr(
        "src.explainability.shap_explainer.ShapExplainer",
        lambda *args, **kwargs: mock_explainer,
    )


def test_train_initial_run_without_champion(
    tmp_path,
    train_mocks,
    monkeypatch,
    sample_df,
):
    config = _training_config(tmp_path)
    monkeypatch.setattr("src.models.train_model.get_config", lambda: config)
    monkeypatch.setattr("src.utils.config.get_config", lambda: config)

    result = train()

    assert result["champion"] in {"logistic_regression", "lightgbm"}
    assert result["decision"]["action"] == "initial"
    registry = ModelRegistry.load(
        config=config,
        registry_path=tmp_path / "models" / "registry.db",
    )
    assert registry.get_current_champion_name() == result["champion"]


def test_train_with_existing_champion(
    tmp_path,
    train_mocks,
    monkeypatch,
    sample_df,
):
    config = _training_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    registry_path = models_dir / "registry.db"

    registry = ModelRegistry(config=config, registry_path=registry_path)
    registry.evaluate_and_decide(
        {
            "model_name": "logistic_regression",
            "metrics": {"roc_auc": 0.82, "recall": 0.75, "accuracy": 0.8, "precision": 0.7, "f1": 0.72},
            "trained_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "model_name": "lightgbm",
            "metrics": {"roc_auc": 0.80, "recall": 0.78, "accuracy": 0.79, "precision": 0.69, "f1": 0.71},
            "trained_at": "2026-01-01T00:00:00+00:00",
        },
    )

    monkeypatch.setattr("src.models.train_model.get_config", lambda: config)
    monkeypatch.setattr("src.utils.config.get_config", lambda: config)

    result = train()

    assert result["champion"] == registry.get_current_champion_name()
    assert result["decision"]["action"] in {"promote", "retain"}
    assert (models_dir / "best_model.pkl").exists()
    assert (models_dir / "registry.db").exists()
