"""Pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("CHURN_API_KEY", "test-secret-key")
os.environ.setdefault("CHURN_DEBUG", "false")
os.environ.setdefault("CHURN_CORS_ORIGINS", "")

_bootstrap_root = Path(__file__).resolve().parents[1]
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from src.utils.paths import ensure_project_imports

PROJECT_ROOT = ensure_project_imports(Path(__file__))

from src.data.data_loader import load_raw_data


@pytest.fixture(autouse=True)
def configure_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API auth environment variables for all tests."""
    monkeypatch.setenv("CHURN_API_KEY", "test-secret-key")
    monkeypatch.setenv("CHURN_DEBUG", "false")
    monkeypatch.setenv("CHURN_CORS_ORIGINS", "")
    get_api_settings = pytest.importorskip("src.utils.api_settings").get_api_settings
    get_api_settings.cache_clear()


@pytest.fixture
def api_client():
    """FastAPI test client with lifespan enabled."""
    import importlib.util

    from fastapi.testclient import TestClient

    module_path = PROJECT_ROOT / "app" / "api" / "fastapi_app.py"
    spec = importlib.util.spec_from_file_location("churn_fastapi_app", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app/api/fastapi_app.py for tests")
    fastapi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fastapi_module)

    with TestClient(fastapi_module.app) as client:
        client.app_module = fastapi_module
        yield client


@pytest.fixture
def api_headers() -> dict[str, str]:
    return {"X-API-Key": "test-secret-key"}


@pytest.fixture
def strict_rate_limits(monkeypatch: pytest.MonkeyPatch, api_client):
    """Lower rate limits for rate-limit tests."""
    monkeypatch.setitem(api_client.app_module.API_RATE_LIMITS, "predict", "2/minute")


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw_data()


@pytest.fixture
def sample_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.head(200).copy()


@pytest.fixture
def sample_customer() -> dict:
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.10,
        "TotalCharges": 1069.20,
    }
