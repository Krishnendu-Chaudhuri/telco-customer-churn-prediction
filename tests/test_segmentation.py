"""Tests for customer segmentation logic."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.retention_engine.segmentation import CustomerSegmentation
from src.utils.config import get_config


@pytest.fixture
def segment_config():
    config = get_config()
    config = json.loads(json.dumps(config))
    config["segmentation"]["n_clusters"] = 5
    config["training"]["random_state"] = 42
    return config


def _make_segmented_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    rows = [
        {
            "tenure": 60,
            "MonthlyCharges": 110.0,
            "clv_estimate": 5000.0,
            "engagement_score": 8.0,
            "churn_probability": 0.1,
        },
        {
            "tenure": 55,
            "MonthlyCharges": 105.0,
            "clv_estimate": 4800.0,
            "engagement_score": 7.5,
            "churn_probability": 0.15,
        },
        {
            "tenure": 8,
            "MonthlyCharges": 95.0,
            "clv_estimate": 900.0,
            "engagement_score": 4.0,
            "churn_probability": 0.8,
        },
        {
            "tenure": 6,
            "MonthlyCharges": 100.0,
            "clv_estimate": 700.0,
            "engagement_score": 3.5,
            "churn_probability": 0.85,
        },
        {
            "tenure": 10,
            "MonthlyCharges": 35.0,
            "clv_estimate": 400.0,
            "engagement_score": 2.0,
            "churn_probability": 0.7,
        },
        {
            "tenure": 9,
            "MonthlyCharges": 30.0,
            "clv_estimate": 350.0,
            "engagement_score": 1.5,
            "churn_probability": 0.75,
        },
        {
            "tenure": 3,
            "MonthlyCharges": 80.0,
            "clv_estimate": 250.0,
            "engagement_score": 2.5,
            "churn_probability": 0.65,
        },
        {
            "tenure": 2,
            "MonthlyCharges": 75.0,
            "clv_estimate": 200.0,
            "engagement_score": 2.0,
            "churn_probability": 0.6,
        },
        {
            "tenure": 30,
            "MonthlyCharges": 60.0,
            "clv_estimate": 2200.0,
            "engagement_score": 5.0,
            "churn_probability": 0.25,
        },
        {
            "tenure": 28,
            "MonthlyCharges": 58.0,
            "clv_estimate": 2100.0,
            "engagement_score": 4.8,
            "churn_probability": 0.2,
        },
    ]
    df = pd.DataFrame(rows)
    churn_probs = pd.Series(df["churn_probability"].values)
    return df, churn_probs


def test_all_five_segment_labels_assigned_uniquely(segment_config):
    df, churn_probs = _make_segmented_dataframe()
    segmenter = CustomerSegmentation(segment_config)
    segmenter.fit_predict(df, churn_probs)

    mapping = segmenter.segment_mapping
    assert len(mapping) == 5
    assert len(set(mapping.values())) == 5
    expected_labels = set(segment_config["segmentation"]["segment_labels"])
    assert set(mapping.values()).issubset(expected_labels)


def test_degenerate_identical_rows(segment_config):
    identical = pd.DataFrame(
        [
            {
                "tenure": 12,
                "MonthlyCharges": 50.0,
                "clv_estimate": 600.0,
                "engagement_score": 3.0,
            }
        ]
        * 50
    )
    churn_probs = pd.Series([0.4] * 50)
    segmenter = CustomerSegmentation(segment_config)
    result = segmenter.fit_predict(identical, churn_probs)

    assert len(segmenter.segment_mapping) == 5
    assert set(segmenter.segment_mapping.keys()) == {0, 1, 2, 3, 4}
    assert "segment" in result.columns


def test_save_load_predict_roundtrip(segment_config, tmp_path: Path):
    df, churn_probs = _make_segmented_dataframe()
    segmenter = CustomerSegmentation(segment_config)
    fitted = segmenter.fit_predict(df, churn_probs)

    model_path = tmp_path / "kmeans_model.pkl"
    mapping_path = tmp_path / "segment_mapping.json"
    segmenter.save(str(model_path), str(mapping_path))

    loaded = CustomerSegmentation.load(str(model_path), str(mapping_path), segment_config)
    predicted = loaded.predict(df, churn_probs)

    assert predicted["segment"].tolist() == fitted["segment"].tolist()


def test_predict_missing_feature_column(segment_config):
    df, churn_probs = _make_segmented_dataframe()
    segmenter = CustomerSegmentation(segment_config)
    segmenter.fit_predict(df, churn_probs)

    reduced = df.drop(columns=["engagement_score"])
    result = segmenter.predict(reduced, churn_probs)

    assert len(result) == len(reduced)
    assert "segment" in result.columns
    assert result["segment"].notna().all()
