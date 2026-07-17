"""Unit tests for train_model stage functions."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelRegistry
from src.models.train_model import (
    _apply_smote,
    _engineer_features,
    _evaluate,
    _promote,
    _train_candidates,
    _validate,
)
from src.utils.config import get_config


class _MockSearch:
    def __init__(self, model: LogisticRegression) -> None:
        self.best_estimator_ = model
        self.best_params_ = {"C": 1.0}
        self.best_score_ = 0.85


def _training_config(tmp_path) -> dict:
    config = get_config()
    return {
        **config,
        "paths": {
            **config["paths"],
            "models_dir": str(tmp_path / "models"),
            "evaluation_dir": str(tmp_path / "models" / "evaluation"),
            "logs_dir": str(tmp_path / "logs"),
            "processed_data": str(tmp_path / "data" / "processed_customers.parquet"),
        },
    }


def test_validate_returns_report(sample_df):
    report = _validate(sample_df, get_config())
    assert isinstance(report, dict)


def test_engineer_features_splits_data(sample_df):
    config = get_config()
    features, target, ids, pipeline, x_train, x_test, y_train, y_test = _engineer_features(
        sample_df,
        config,
    )
    assert not features.empty
    assert target is not None
    assert len(x_train) + len(x_test) == len(features)
    assert len(y_train) == len(x_train)
    assert pipeline is not None


def test_apply_smote_resamples_when_enabled(sample_df):
    config = get_config()
    _, target, _, _, x_train, _, y_train, _ = _engineer_features(sample_df, config)
    config["training"]["smote"]["enabled"] = True
    balanced_x, balanced_y = _apply_smote(x_train, y_train, config)
    assert len(balanced_x) >= len(x_train)
    assert len(balanced_y) == len(balanced_x)


def test_train_candidates_returns_model_results(sample_df, monkeypatch):
    config = get_config()
    _, _, _, _, x_train, x_test, y_train, y_test = _engineer_features(sample_df, config)

    def mock_tune(x_train_in, y_train_in, config=None):
        lr = LogisticRegression(max_iter=200).fit(x_train_in, y_train_in)
        lgbm = LogisticRegression(max_iter=200).fit(x_train_in, y_train_in)
        return {
            "logistic_regression": _MockSearch(lr),
            "lightgbm": _MockSearch(lgbm),
        }

    monkeypatch.setattr("src.models.train_model.tune_models", mock_tune)
    tuned_models, comparison, model_results, comparison_df, trained_at = _train_candidates(
        x_train,
        y_train,
        x_test,
        y_test,
        config,
    )
    assert set(tuned_models) == {"logistic_regression", "lightgbm"}
    assert "roc_auc" in comparison["logistic_regression"]
    assert "logistic_regression" in model_results
    assert not comparison_df.empty
    assert trained_at


def test_promote_initial_run(tmp_path):
    config = _training_config(tmp_path)
    staging_registry = tmp_path / "staging" / "registry.db"
    staging_registry.parent.mkdir(parents=True, exist_ok=True)

    model_results = {
        "logistic_regression": {
            "model_name": "logistic_regression",
            "metrics": {"roc_auc": 0.84, "recall": 0.75, "accuracy": 0.8, "precision": 0.7, "f1": 0.72},
            "trained_at": "2026-01-01T00:00:00+00:00",
        },
        "lightgbm": {
            "model_name": "lightgbm",
            "metrics": {"roc_auc": 0.82, "recall": 0.78, "accuracy": 0.79, "precision": 0.69, "f1": 0.71},
            "trained_at": "2026-01-01T00:00:00+00:00",
        },
    }

    from src.utils.paths import ProjectPaths

    paths = ProjectPaths(config)
    decision, champion_name, registry = _promote(
        model_results,
        paths,
        staging_registry,
        config,
    )
    assert decision["action"] == "initial"
    assert champion_name in {"logistic_regression", "lightgbm"}
    assert registry.get_current_champion_name() == champion_name


def test_promote_with_existing_champion(tmp_path):
    config = _training_config(tmp_path)
    staging_registry = tmp_path / "staging" / "registry.db"
    staging_registry.parent.mkdir(parents=True, exist_ok=True)

    from src.utils.paths import ProjectPaths

    paths = ProjectPaths(config)
    live_registry = ModelRegistry(config=config, registry_path=paths.registry_db)
    live_registry.evaluate_and_decide(
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

    model_results = {
        "logistic_regression": {
            "model_name": "logistic_regression",
            "metrics": {"roc_auc": 0.83, "recall": 0.74, "accuracy": 0.8, "precision": 0.7, "f1": 0.72},
            "trained_at": "2026-01-02T00:00:00+00:00",
        },
        "lightgbm": {
            "model_name": "lightgbm",
            "metrics": {"roc_auc": 0.84, "recall": 0.72, "accuracy": 0.79, "precision": 0.69, "f1": 0.71},
            "trained_at": "2026-01-02T00:00:00+00:00",
        },
    }

    decision, champion_name, registry = _promote(
        model_results,
        paths,
        staging_registry,
        config,
    )
    assert decision["action"] in {"promote", "retain"}
    assert registry.get_current_champion_name() == champion_name


def test_evaluate_writes_metrics(tmp_path, sample_df):
    config = get_config()
    features, _, _, _, x_train, x_test, y_train, y_test = _engineer_features(sample_df, config)
    model = LogisticRegression(max_iter=200).fit(x_train, y_train)
    staging = {"evaluation_dir": tmp_path / "evaluation"}
    staging["evaluation_dir"].mkdir(parents=True, exist_ok=True)

    eval_results = _evaluate(model, x_test, y_test, features, staging, "logistic_regression")
    assert "metrics" in eval_results
    assert "roc_auc" in eval_results["metrics"]
