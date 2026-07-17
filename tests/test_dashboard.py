"""Streamlit dashboard smoke tests and logic unit tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.dashboard.logic import (
    HIGH_RISK_THRESHOLD,
    build_segment_clv_summary,
    compute_executive_kpis,
    format_champion_panel,
    segmentation_has_data,
)
from src.models.registry import ModelRegistry, seed_from_json_payload

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = PROJECT_ROOT / "app" / "dashboard" / "streamlit_app.py"

PAGES = [
    "Executive Dashboard",
    "Customer Lookup",
    "Segmentation",
    "Model Performance",
    "Explainability",
]


@pytest.fixture
def dashboard_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return sample_df.copy()


def _run_page(page_name: str, dashboard_df: pd.DataFrame, monkeypatch) -> AppTest:
    import app.dashboard.streamlit_app as dashboard_module

    monkeypatch.setattr(dashboard_module, "load_predictor", lambda: None)
    if hasattr(dashboard_module.load_predictor, "clear"):
        dashboard_module.load_predictor.clear()
    monkeypatch.setattr(dashboard_module, "load_data", lambda: dashboard_df)

    app = AppTest.from_file(str(DASHBOARD_PATH))
    app.run(timeout=30)
    if app.sidebar.radio:
        app.sidebar.radio[0].set_value(page_name).run(timeout=30)
    return app


@pytest.mark.parametrize("page_name", PAGES)
def test_dashboard_page_smoke(page_name: str, dashboard_df, monkeypatch):
    app = _run_page(page_name, dashboard_df, monkeypatch)
    assert not app.exception


def test_compute_executive_kpis_from_churn_column(sample_df):
    kpis = compute_executive_kpis(sample_df)
    assert 0 <= kpis["churn_rate"] <= 1
    assert kpis["high_risk_count"] == 0
    assert kpis["revenue_at_risk"] == 0.0
    assert kpis["avg_clv"] == 0.0


def test_compute_executive_kpis_with_probability_columns():
    df = pd.DataFrame(
        {
            "churn_probability": [0.8, 0.3, 0.75, 0.1],
            "clv_estimate": [1000.0, 500.0, 800.0, 200.0],
        }
    )
    kpis = compute_executive_kpis(df)
    assert kpis["churn_rate"] == pytest.approx(0.4875)
    assert kpis["revenue_at_risk"] == pytest.approx(800.0 + 150.0 + 600.0 + 20.0)
    assert kpis["high_risk_count"] == 2
    assert kpis["avg_clv"] == pytest.approx(625.0)


def test_compute_executive_kpis_high_risk_threshold():
    df = pd.DataFrame({"churn_probability": [0.69, HIGH_RISK_THRESHOLD, 0.95]})
    kpis = compute_executive_kpis(df)
    assert kpis["high_risk_count"] == 2


def test_compute_executive_kpis_with_predictor(sample_df):
    predictor = MagicMock()
    predictor.predict_proba.return_value = pd.Series([0.5, 0.6], index=sample_df.head(2).index)
    predictor.pipeline.cleaner.transform.return_value = sample_df.head(2)
    predictor.pipeline.feature_engineer.transform.return_value = pd.DataFrame(
        {"clv_estimate": [100.0, 200.0]},
        index=sample_df.head(2).index,
    )

    kpis = compute_executive_kpis(sample_df.head(2), predictor)
    assert kpis["revenue_at_risk"] == pytest.approx(50.0 + 120.0)


def test_segmentation_has_data_true():
    df = pd.DataFrame({"segment": ["A", "B"]})
    assert segmentation_has_data(df) is True


def test_segmentation_has_data_false_missing_column():
    df = pd.DataFrame({"customerID": ["A", "B"]})
    assert segmentation_has_data(df) is False


def test_segmentation_has_data_false_empty():
    df = pd.DataFrame({"segment": []})
    assert segmentation_has_data(df) is False


def test_build_segment_clv_summary_returns_aggregates():
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "B"],
            "clv_estimate": [100.0, 200.0, 50.0],
        }
    )
    summary = build_segment_clv_summary(df)
    assert summary is not None
    assert set(summary["segment"]) == {"A", "B"}
    assert summary.loc[summary["segment"] == "A", "clv_estimate"].iloc[0] == pytest.approx(300.0)


def test_build_segment_clv_summary_none_without_segment():
    df = pd.DataFrame({"clv_estimate": [100.0]})
    assert build_segment_clv_summary(df) is None


def test_build_segment_clv_summary_none_without_clv():
    df = pd.DataFrame({"segment": ["A"]})
    assert build_segment_clv_summary(df) is None


def _registry_payload() -> dict:
    return {
        "champion": {
            "model_name": "logistic_regression",
            "metrics": {"roc_auc": 0.84, "recall": 0.75},
            "trained_at": "2026-01-01T00:00:00+00:00",
            "promoted_at": "2026-01-01T00:00:00+00:00",
        },
        "challenger": {
            "model_name": "lightgbm",
            "metrics": {"roc_auc": 0.82, "recall": 0.78},
            "trained_at": "2026-01-01T00:00:00+00:00",
            "promoted_at": None,
        },
        "promotion_threshold": 0.005,
        "history": [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "action": "initial",
                "previous_champion": None,
                "new_champion": "logistic_regression",
                "champion_metric": 0.84,
                "challenger_metric": 0.82,
                "delta": -0.02,
            }
        ],
    }


def test_format_champion_panel_assembles_roles(tmp_path):
    registry = ModelRegistry(registry_path=tmp_path / "registry.db")
    seed_from_json_payload(registry, _registry_payload())
    panel = format_champion_panel(registry)

    assert panel["champion_model"] == "logistic_regression"
    assert panel["challenger_model"] == "lightgbm"
    assert panel["promotion_threshold"] == 0.005
    assert len(panel["metric_rows"]) == 2
    assert panel["metric_rows"][0]["roc_auc"] == 0.84
    assert len(panel["history"]) == 1


def test_format_champion_panel_empty_registry(tmp_path):
    registry = ModelRegistry(registry_path=tmp_path / "empty.db")
    panel = format_champion_panel(registry)

    assert panel["champion_model"] is None
    assert panel["challenger_model"] is None
    assert panel["history"] == []
