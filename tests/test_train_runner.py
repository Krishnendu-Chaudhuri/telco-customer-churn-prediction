"""Tests for supervised training subprocess worker."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from src.models import train_runner


@pytest.fixture
def training_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / ".training.lock"
    monkeypatch.setattr(train_runner, "training_lock_path", lambda: lock_path)
    return lock_path


def test_acquire_and_release_lock(training_lock):
    train_runner.acquire_lock()
    assert training_lock.exists()
    payload = json.loads(training_lock.read_text(encoding="utf-8"))
    assert "pid" in payload
    assert "started_at" in payload
    train_runner.release_lock()
    assert not training_lock.exists()


def test_clear_stale_lock_removes_dead_pid(training_lock, monkeypatch):
    training_lock.write_text(
        json.dumps({"pid": 999999, "started_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_runner, "pid_alive", lambda pid: False)
    assert train_runner.clear_stale_lock() is True
    assert not training_lock.exists()


def test_lock_is_active_when_pid_alive(training_lock, monkeypatch):
    training_lock.write_text(
        json.dumps({"pid": 1234, "started_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_runner, "pid_alive", lambda pid: True)
    assert train_runner.lock_is_active() is True


def test_poll_training_success_updates_status(training_lock, monkeypatch):
    app = FastAPI()
    app.state.lock = __import__("threading").Lock()
    app.state.predictor = None
    app.state.training_status = {
        "status": "idle",
        "phase": "idle",
        "last_trained_at": None,
        "message": "",
    }

    process = MagicMock()
    process.poll.side_effect = [None, 0]
    process.returncode = 0
    process.stderr = None

    mock_predictor = MagicMock()
    monkeypatch.setattr(
        "src.models.predictor.ChurnPredictor",
        lambda config: MagicMock(load=lambda: mock_predictor),
    )

    train_runner.acquire_lock()
    train_runner.poll_training(app, process)

    assert app.state.training_status["status"] == "completed"
    assert app.state.predictor is mock_predictor
    assert not training_lock.exists()


def test_poll_training_failure_clears_lock(training_lock):
    app = FastAPI()
    app.state.lock = __import__("threading").Lock()
    app.state.predictor = None
    app.state.training_status = {
        "status": "idle",
        "phase": "idle",
        "last_trained_at": None,
        "message": "",
    }

    process = MagicMock()
    process.poll.return_value = 1
    process.returncode = 1
    process.stderr = MagicMock(read=lambda: "boom")

    train_runner.acquire_lock()
    train_runner.poll_training(app, process)

    assert app.state.training_status["status"] == "failed"
    assert not training_lock.exists()


def test_trigger_training_returns_409_when_lock_active(api_client, api_headers, monkeypatch):
    monkeypatch.setattr(train_runner, "lock_is_active", lambda: True)
    response = api_client.post("/v1/train", headers=api_headers)
    assert response.status_code == 409
