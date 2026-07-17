"""FastAPI application for churn prediction serving."""

import hashlib
import hmac
import json
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_bootstrap_root = Path(__file__).resolve().parents[2]
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from src.utils.paths import ensure_project_imports

PROJECT_ROOT = ensure_project_imports(Path(__file__))

from src.utils.api_keys import SCOPE_ADMIN, has_active_keys, key_has_scope, verify_key
from src.utils.api_settings import get_api_settings
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.logging_filters import redact_payload
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)
config = get_config()
paths = ProjectPaths(config)

API_RATE_LIMITS: dict[str, str] = config.get("api", {}).get(
    "rate_limits",
    {
        "predict": "60/minute",
        "predict_batch": "10/minute",
        "train": "1/minute",
    },
)

limiter = Limiter(key_func=get_remote_address)

PROTECTED_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"description": "Invalid or missing API key"},
    403: {"description": "Insufficient API key scope"},
    503: {"description": "Service unavailable"},
    500: {"description": "Internal server error"},
}


def _default_training_status() -> dict[str, Any]:
    return {
        "status": "idle",
        "phase": "idle",
        "last_trained_at": None,
        "message": "",
    }


def _client_error_detail(exc: Exception) -> str:
    """Return sanitized or detailed error message for API clients."""
    if get_api_settings().CHURN_DEBUG:
        return str(exc)
    return "An internal error occurred."


def _sanitize_training_status(status: dict[str, Any]) -> dict[str, Any]:
    """Expose only non-sensitive training status fields."""
    return {
        "status": status.get("status", "idle"),
        "phase": status.get("phase", "idle"),
        "last_trained_at": status.get("last_trained_at"),
    }


def _auth_configured() -> bool:
    """Return True when env or per-client API keys are configured."""
    return bool(get_api_settings().CHURN_API_KEY) or has_active_keys()


def _is_valid_api_key(x_api_key: str | None) -> bool:
    expected_key = get_api_settings().CHURN_API_KEY
    if expected_key and x_api_key and hmac.compare_digest(x_api_key, expected_key):
        return True
    return verify_key(x_api_key)


def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Validate API key for protected endpoints."""
    if not _auth_configured():
        raise HTTPException(status_code=503, detail="API key authentication is not configured")
    if not _is_valid_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def verify_admin_scope(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Validate API key and require admin scope for privileged endpoints."""
    verify_api_key(x_api_key)
    if not key_has_scope(x_api_key, SCOPE_ADMIN):
        raise HTTPException(status_code=403, detail="Admin scope required for this endpoint")


def _artifacts_ready() -> bool:
    """Return True when serving artifacts are available."""
    from src.models.predictor import ChurnPredictor

    return ChurnPredictor(config).is_ready


def _reload_predictor_if_ready(request: Request) -> None:
    """Reload cached predictor from disk when artifacts are available."""
    if not _artifacts_ready():
        logger.warning("Skipping predictor reload: artifacts not ready")
        return

    from src.models.predictor import ChurnPredictor

    with request.app.state.lock:
        previous = request.app.state.predictor
        try:
            request.app.state.predictor = ChurnPredictor(config).load()
        except FileNotFoundError as exc:
            logger.warning(
                "Predictor reload failed after registry change; preserving existing instance: %s",
                exc,
            )
            request.app.state.predictor = previous


def _get_served_by_model() -> str | None:
    """Return the current champion model name from registry or metadata."""
    if paths.registry_db.exists():
        from src.models.registry import ModelRegistry

        registry = ModelRegistry.load(config=config, registry_path=paths.registry_db)
        return registry.get_current_champion_name()
    if paths.model_metadata.exists():
        with paths.model_metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata.get("best_model_name")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    app.state.lock = threading.Lock()
    app.state.predictor = None
    app.state.training_status = _default_training_status()
    try:
        if _artifacts_ready():
            from src.models.predictor import ChurnPredictor

            with app.state.lock:
                app.state.predictor = ChurnPredictor(config).load()
            logger.info("Model artifacts loaded at startup")
    except Exception as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Telco Churn & Retention API",
    description="Production API for churn prediction and retention recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_api_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

Instrumentator().instrument(app)

v1_router = APIRouter(prefix="/v1")


@app.get("/metrics", tags=["Observability"], responses=PROTECTED_RESPONSES)
def metrics(_: Annotated[None, Depends(verify_api_key)] = None) -> Response:
    """Prometheus metrics (requires API key)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class CustomerInput(BaseModel):
    customerID: str | None = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ChallengerPrediction(BaseModel):
    model_name: str
    churn_probability: float
    prediction: int


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    retention_strategy: str
    clv: float
    customerID: str | None = None
    served_by_model: str | None = None
    retention_cost: float | None = None
    expected_revenue_saved: float | None = None
    explanation: str | None = None
    top_contributors: list[dict[str, Any]] | None = None
    challenger_prediction: ChallengerPrediction | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput]
    include_explanation: bool = False
    shadow: bool = False


class ChampionRole(BaseModel):
    model_name: str | None = None
    metrics: dict[str, float] | None = None
    trained_at: str | None = None
    promoted_at: str | None = None


class ChampionHistoryEntry(BaseModel):
    timestamp: str
    action: str
    previous_champion: str | None = None
    new_champion: str
    champion_metric: float
    challenger_metric: float
    delta: float


class ChampionStatusResponse(BaseModel):
    champion: ChampionRole | None = None
    challenger: ChampionRole | None = None
    promotion_threshold: float
    history: list[ChampionHistoryEntry]


class RollbackResponse(BaseModel):
    action: str
    champion: str
    challenger: str
    delta: float


for _model in (
    ChampionRole,
    ChampionHistoryEntry,
    ChampionStatusResponse,
    RollbackResponse,
    ChallengerPrediction,
    PredictionResponse,
    BatchPredictionResponse,
):
    _model.model_rebuild(_types_namespace=globals())


def _artifact_checksum() -> str | None:
    if not paths.model_metadata.exists():
        return None
    content = paths.model_metadata.read_bytes()
    return hashlib.md5(content).hexdigest()


def _get_predictor(request: Request) -> Any:
    with request.app.state.lock:
        if request.app.state.predictor is None:
            from src.models.predictor import ChurnPredictor

            try:
                request.app.state.predictor = ChurnPredictor(config).load()
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Model artifacts not found. Run training via POST /v1/train or `python src/models/train_model.py`.",
                ) from exc
        return request.app.state.predictor


def _load_registry_status(history_limit: int = 10) -> ChampionStatusResponse:
    from src.models.registry import ModelRegistry

    if not paths.registry_db.exists():
        raise HTTPException(
            status_code=404,
            detail="Champion/challenger registry not found. Run training first.",
        )

    registry = ModelRegistry.load(config=config, registry_path=paths.registry_db)
    history = registry.history[-history_limit:]
    return ChampionStatusResponse(
        champion=ChampionRole(**registry.champion) if registry.champion else None,
        challenger=ChampionRole(**registry.challenger) if registry.challenger else None,
        promotion_threshold=registry.promotion_threshold,
        history=[ChampionHistoryEntry(**entry) for entry in history],
    )


@app.get("/health")
def health(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> dict[str, Any]:
    """Health check endpoint (minimal public payload; extended with valid API key)."""
    with request.app.state.lock:
        predictor = request.app.state.predictor
        training_status = request.app.state.training_status
        model_loaded = predictor is not None and getattr(predictor, "is_ready", False)
        sanitized_status = _sanitize_training_status(training_status)

    response: dict[str, Any] = {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if _is_valid_api_key(x_api_key):
        metadata: dict[str, Any] = {}
        if paths.model_metadata.exists():
            with paths.model_metadata.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        response.update(
            {
                "model_version": _get_served_by_model() or metadata.get("best_model_name"),
                "served_by_model": _get_served_by_model(),
                "trained_at": metadata.get("trained_at"),
                "artifact_checksum": _artifact_checksum(),
                "training_status": sanitized_status,
            }
        )

    return response


@v1_router.get(
    "/champion/status", response_model=ChampionStatusResponse, responses=PROTECTED_RESPONSES
)
def champion_status(
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> ChampionStatusResponse:
    """Return current champion/challenger state and recent promotion history."""
    return _load_registry_status()


@v1_router.post(
    "/champion/rollback", response_model=RollbackResponse, responses=PROTECTED_RESPONSES
)
def champion_rollback(
    request: Request,
    _: Annotated[None, Depends(verify_admin_scope)] = None,
) -> RollbackResponse:
    """Manually swap champion and challenger without retraining."""
    from src.models.registry import ModelRegistry

    if not paths.registry_db.exists():
        raise HTTPException(
            status_code=404,
            detail="Champion/challenger registry not found. Run training first.",
        )

    registry = ModelRegistry.load(config=config, registry_path=paths.registry_db)
    try:
        result = registry.rollback()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.warning(
        "Manual champion rollback executed: champion=%s challenger=%s",
        result["champion"],
        result["challenger"],
    )

    _reload_predictor_if_ready(request)

    return RollbackResponse(
        action=result["action"],
        champion=result["champion"],
        challenger=result["challenger"],
        delta=result["delta"],
    )


@v1_router.post("/train", responses=PROTECTED_RESPONSES)
@limiter.limit(lambda: API_RATE_LIMITS.get("train", "1/minute"))
def trigger_training(
    request: Request,
    _: Annotated[None, Depends(verify_admin_scope)] = None,
) -> dict[str, str]:
    """Trigger model training in a supervised subprocess."""
    from src.models.train_runner import clear_stale_lock, launch_training, lock_is_active

    with request.app.state.lock:
        if request.app.state.training_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Training already in progress")
        if lock_is_active():
            raise HTTPException(status_code=409, detail="Training already in progress")
        clear_stale_lock()
        request.app.state.training_status = {
            "status": "running",
            "phase": "training",
            "last_trained_at": None,
            "message": "Training queued",
        }

    try:
        launch_training(request.app)
    except Exception as exc:
        with request.app.state.lock:
            request.app.state.training_status = {
                "status": "failed",
                "phase": "failed",
                "last_trained_at": None,
                "message": _client_error_detail(exc),
            }
        from src.models.train_runner import release_lock

        release_lock()
        raise HTTPException(status_code=500, detail=_client_error_detail(exc)) from exc

    return {"message": "Training started in background"}


@v1_router.post("/predict", response_model=PredictionResponse, responses=PROTECTED_RESPONSES)
@limiter.limit(lambda: API_RATE_LIMITS.get("predict", "60/minute"))
def predict(
    request: Request,
    customer: CustomerInput = Body(),
    include_explanation: bool = False,
    shadow: bool = False,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> PredictionResponse:
    """Predict churn for a single customer."""
    if not _artifacts_ready():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run training via POST /v1/train or `python src/models/train_model.py`.",
        )

    try:
        churn_predictor = _get_predictor(request)
        result = churn_predictor.predict_single(
            customer.model_dump(),
            include_explanation=include_explanation,
            shadow=shadow,
        )
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception(
            "Prediction failed for customer payload: %s",
            redact_payload(customer.model_dump()),
        )
        raise HTTPException(status_code=500, detail=_client_error_detail(exc)) from exc


@v1_router.post(
    "/predict_batch", response_model=BatchPredictionResponse, responses=PROTECTED_RESPONSES
)
@limiter.limit(lambda: API_RATE_LIMITS.get("predict_batch", "10/minute"))
def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest = Body(),
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> BatchPredictionResponse:
    """Predict churn for multiple customers."""
    if not _artifacts_ready():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run training via POST /v1/train or `python src/models/train_model.py`.",
        )

    try:
        churn_predictor = _get_predictor(request)
        df = pd.DataFrame([item.model_dump() for item in batch_request.customers])
        predictions = churn_predictor.predict_batch(
            df,
            include_explanation=batch_request.include_explanation,
            shadow=batch_request.shadow,
        )
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=[PredictionResponse(**item) for item in predictions],
        )
    except Exception as exc:
        logger.exception(
            "Batch prediction failed for %s customers",
            len(batch_request.customers),
        )
        raise HTTPException(status_code=500, detail=_client_error_detail(exc)) from exc


app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
