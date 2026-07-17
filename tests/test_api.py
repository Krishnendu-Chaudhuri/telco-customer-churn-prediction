from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.registry import ModelRegistry, seed_from_json_payload


def _sample_registry_payload() -> dict:
    return {
        "champion": {
            "model_name": "logistic_regression",
            "metrics": {"roc_auc": 0.839, "recall": 0.75},
            "trained_at": "2026-06-10T12:00:00+00:00",
            "promoted_at": "2026-06-10T12:00:00+00:00",
        },
        "challenger": {
            "model_name": "lightgbm",
            "metrics": {"roc_auc": 0.835, "recall": 0.80},
            "trained_at": "2026-06-10T12:00:00+00:00",
            "promoted_at": None,
        },
        "promotion_threshold": 0.005,
        "history": [
            {
                "timestamp": "2026-06-10T12:00:00+00:00",
                "action": "initial",
                "previous_champion": None,
                "new_champion": "logistic_regression",
                "champion_metric": 0.839,
                "challenger_metric": 0.835,
                "delta": -0.004,
            }
        ],
    }


@pytest.fixture
def registry_file(tmp_path):
    path = tmp_path / "registry.db"
    registry = ModelRegistry(registry_path=path)
    seed_from_json_payload(registry, _sample_registry_payload())
    registry.save()
    return path


def test_health_endpoint_public_minimal(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_loaded" in payload
    assert "timestamp" in payload
    assert "training_status" not in payload
    assert "artifact_checksum" not in payload


def test_health_endpoint_authenticated_extended(api_client, api_headers):
    response = api_client.get("/health", headers=api_headers)
    assert response.status_code == 200
    payload = response.json()
    assert "training_status" in payload
    assert "status" in payload["training_status"]
    assert "message" not in payload["training_status"]


def test_predict_without_api_key_returns_401(api_client, sample_customer):
    response = api_client.post("/v1/predict", json=sample_customer)
    assert response.status_code == 401


def test_predict_with_valid_api_key(api_client, api_headers, sample_customer):
    response = api_client.post("/v1/predict", json=sample_customer, headers=api_headers)
    assert response.status_code in {200, 503}


def test_rate_limit_returns_429(api_client, api_headers, sample_customer, strict_rate_limits):
    first = api_client.post("/v1/predict", json=sample_customer, headers=api_headers)
    second = api_client.post("/v1/predict", json=sample_customer, headers=api_headers)
    third = api_client.post("/v1/predict", json=sample_customer, headers=api_headers)

    status_codes = {first.status_code, second.status_code, third.status_code}
    assert 429 in status_codes or first.status_code == 503


def test_champion_status_returns_registry(api_client, api_headers, registry_file, monkeypatch):
    monkeypatch.setattr(api_client.app_module.paths, "registry_db", registry_file)

    response = api_client.get("/v1/champion/status", headers=api_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["champion"]["model_name"] == "logistic_regression"
    assert payload["challenger"]["model_name"] == "lightgbm"
    assert payload["promotion_threshold"] == 0.005
    assert len(payload["history"]) == 1
    assert payload["history"][0]["action"] == "initial"


def test_champion_status_requires_auth(api_client, registry_file, monkeypatch):
    monkeypatch.setattr(api_client.app_module.paths, "registry_db", registry_file)

    response = api_client.get("/v1/champion/status")

    assert response.status_code == 401


def test_champion_rollback_swaps_roles(
    api_client,
    api_headers,
    registry_file,
    monkeypatch,
):
    monkeypatch.setattr(api_client.app_module.paths, "registry_db", registry_file)
    monkeypatch.setattr(api_client.app_module, "_artifacts_ready", lambda: True)
    with api_client.app.state.lock:
        api_client.app.state.predictor = MagicMock(is_ready=True)

    rollback = api_client.post("/v1/champion/rollback", headers=api_headers)
    assert rollback.status_code == 200
    rollback_payload = rollback.json()
    assert rollback_payload["champion"] == "lightgbm"
    assert rollback_payload["challenger"] == "logistic_regression"
    assert rollback_payload["action"] == "manual_rollback"

    status = api_client.get("/v1/champion/status", headers=api_headers)
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["champion"]["model_name"] == "lightgbm"
    assert status_payload["challenger"]["model_name"] == "logistic_regression"
    assert status_payload["history"][-1]["action"] == "manual_rollback"


def test_predict_with_shadow_includes_challenger_prediction(
    api_client,
    api_headers,
    sample_customer,
    monkeypatch,
):
    mock_predictor = MagicMock()
    mock_predictor.is_ready = True
    mock_predictor.predict_single.return_value = {
        "churn_probability": 0.82,
        "risk_level": "high",
        "retention_strategy": "Discount offer",
        "clv": 1200.0,
        "served_by_model": "logistic_regression",
        "challenger_prediction": {
            "model_name": "lightgbm",
            "churn_probability": 0.79,
            "prediction": 1,
        },
    }

    monkeypatch.setattr(api_client.app_module, "_artifacts_ready", lambda: True)
    monkeypatch.setattr(
        api_client.app_module,
        "_get_predictor",
        lambda request: mock_predictor,
    )

    response = api_client.post(
        "/v1/predict",
        params={"shadow": "true"},
        json=sample_customer,
        headers=api_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["served_by_model"] == "logistic_regression"
    assert payload["churn_probability"] == 0.82
    assert payload["challenger_prediction"]["model_name"] == "lightgbm"
    mock_predictor.predict_single.assert_called_once()
    assert mock_predictor.predict_single.call_args.kwargs["shadow"] is True
