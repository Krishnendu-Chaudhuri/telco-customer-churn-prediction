"""Tests for per-client API key authentication."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.api_keys import issue_key, revoke_by_prefix


@pytest.fixture
def client_db(tmp_path, monkeypatch):
    db_path = tmp_path / "registry.db"

    def _db_path():
        return db_path

    monkeypatch.setattr("src.utils.api_keys._db_path", _db_path)
    return db_path


@pytest.fixture
def mock_predictor(api_client, monkeypatch):
    mock = MagicMock()
    mock.predict_single.return_value = {
        "churn_probability": 0.5,
        "risk_level": "medium",
        "retention_strategy": "Check-in",
        "clv": 100.0,
    }
    monkeypatch.setattr(api_client.app_module, "_artifacts_ready", lambda: True)
    monkeypatch.setattr(api_client.app_module, "_get_predictor", lambda request: mock)
    return mock


def test_issued_key_authenticates(
    api_client,
    sample_customer,
    client_db,
    mock_predictor,
):
    issued = issue_key("test-client")
    response = api_client.post(
        "/v1/predict",
        json=sample_customer,
        headers={"X-API-Key": issued},
    )
    assert response.status_code == 200


def test_predict_scope_denied_on_train(api_client, client_db, monkeypatch):
    issued = issue_key("predict-only", scopes="predict")
    monkeypatch.setattr(
        "src.models.train_runner.launch_training",
        lambda app: MagicMock(),
    )
    response = api_client.post("/v1/train", headers={"X-API-Key": issued})
    assert response.status_code == 403
    assert "Admin scope required" in response.json()["detail"]


def test_predict_scope_denied_on_rollback(api_client, client_db):
    issued = issue_key("predict-only", scopes="predict")
    response = api_client.post("/v1/champion/rollback", headers={"X-API-Key": issued})
    assert response.status_code == 403


def test_admin_scope_can_trigger_train(api_client, client_db, monkeypatch):
    issued = issue_key("ops-team", scopes="admin")
    monkeypatch.setattr(
        "src.models.train_runner.launch_training",
        lambda app: MagicMock(),
    )
    monkeypatch.setattr("src.models.train_runner.lock_is_active", lambda: False)
    monkeypatch.setattr("src.models.train_runner.clear_stale_lock", lambda: False)

    response = api_client.post("/v1/train", headers={"X-API-Key": issued})
    assert response.status_code == 200


def test_revoked_key_returns_401(
    api_client,
    sample_customer,
    client_db,
):
    issued = issue_key("test-client")
    revoke_by_prefix(issued[:8])

    response = api_client.post(
        "/v1/predict",
        json=sample_customer,
        headers={"X-API-Key": issued},
    )
    assert response.status_code == 401


def test_unknown_key_returns_401(api_client, sample_customer, client_db):
    response = api_client.post(
        "/v1/predict",
        json=sample_customer,
        headers={"X-API-Key": "totally-unknown-key-value"},
    )
    assert response.status_code == 401


def test_env_admin_key_still_works(
    api_client,
    sample_customer,
    api_headers,
    mock_predictor,
):
    response = api_client.post(
        "/v1/predict",
        json=sample_customer,
        headers=api_headers,
    )
    assert response.status_code == 200


def test_env_admin_key_can_trigger_train(api_client, api_headers, monkeypatch):
    monkeypatch.setattr(
        "src.models.train_runner.launch_training",
        lambda app: MagicMock(),
    )
    monkeypatch.setattr("src.models.train_runner.lock_is_active", lambda: False)
    monkeypatch.setattr("src.models.train_runner.clear_stale_lock", lambda: False)

    response = api_client.post("/v1/train", headers=api_headers)
    assert response.status_code == 200
