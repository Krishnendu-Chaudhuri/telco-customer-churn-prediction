"""Supervised subprocess training worker with PID lock file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths, get_project_root

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 2.0


def training_lock_path() -> Path:
    return ProjectPaths(get_config()).training_lock


def _read_lock_payload() -> dict[str, Any] | None:
    lock_path = training_lock_path()
    if not lock_path.exists():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def pid_alive(pid: int) -> bool:
    """Return True when a process id is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def clear_stale_lock() -> bool:
    """Remove a lock file when its PID is no longer alive."""
    payload = _read_lock_payload()
    if payload is None:
        return False
    pid = int(payload.get("pid", 0))
    if pid_alive(pid):
        return False
    release_lock()
    return True


def acquire_lock(pid: int | None = None) -> None:
    """Write a process PID and start timestamp to the lock file."""
    lock_path = training_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid if pid is not None else os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")


def release_lock() -> None:
    """Remove the training lock file if present."""
    lock_path = training_lock_path()
    if lock_path.exists():
        lock_path.unlink(missing_ok=True)


def lock_is_active() -> bool:
    """Return True when a live training process holds the lock."""
    payload = _read_lock_payload()
    if payload is None:
        return False
    return pid_alive(int(payload.get("pid", 0)))


def start_training_subprocess() -> subprocess.Popen[str]:
    """Launch training in a detached subprocess."""
    project_root = get_project_root()
    return subprocess.Popen(
        [sys.executable, "-m", "src.models.train_model"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _set_training_status(app: FastAPI, status: dict[str, Any]) -> None:
    with app.state.lock:
        app.state.training_status = status


def _mark_running(app: FastAPI, message: str, phase: str = "training") -> None:
    _set_training_status(
        app,
        {
            "status": "running",
            "phase": phase,
            "last_trained_at": None,
            "message": message,
        },
    )


def _client_error_detail(exc: Exception) -> str:
    from src.utils.api_settings import get_api_settings

    if get_api_settings().CHURN_DEBUG:
        return str(exc)
    return "An internal error occurred."


def poll_training(app: FastAPI, process: subprocess.Popen[str]) -> None:
    """Poll subprocess completion and update app training status."""
    try:
        _mark_running(app, "Training subprocess started")
        while process.poll() is None:
            time.sleep(POLL_INTERVAL_SECONDS)

        if process.returncode == 0:
            from src.models.predictor import ChurnPredictor

            config = get_config()
            with app.state.lock:
                app.state.predictor = ChurnPredictor(config).load()
                app.state.training_status = {
                    "status": "completed",
                    "phase": "published",
                    "last_trained_at": datetime.now(timezone.utc).isoformat(),
                    "message": "Training subprocess completed successfully",
                }
            logger.info("Training subprocess completed successfully")
        else:
            stderr = process.stderr.read() if process.stderr is not None else ""
            detail = stderr.strip() or f"Training subprocess failed with code {process.returncode}"
            _set_training_status(
                app,
                {
                    "status": "failed",
                    "phase": "failed",
                    "last_trained_at": None,
                    "message": detail,
                },
            )
            logger.error("Training subprocess failed: %s", detail)
    except Exception as exc:
        logger.exception("Training poll thread failed")
        _set_training_status(
            app,
            {
                "status": "failed",
                "phase": "failed",
                "last_trained_at": None,
                "message": _client_error_detail(exc),
            },
        )
    finally:
        release_lock()


def launch_training(app: FastAPI) -> subprocess.Popen[str]:
    """Start subprocess, record its PID in the lock, and schedule polling."""
    process = start_training_subprocess()
    acquire_lock(process.pid)
    thread = threading.Thread(
        target=poll_training,
        args=(app, process),
        name="training-poll",
        daemon=True,
    )
    thread.start()
    return process
