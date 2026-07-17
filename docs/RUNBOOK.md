# Operational Runbook

Command-oriented guide for on-call response. Paths and endpoints match the current codebase.

## Training failure

### Lock file

Background training uses a PID lock at:

```
models/.training.lock
```

Resolved via `ProjectPaths.training_lock` (`src/utils/paths.py`) and `train_runner.training_lock_path()` (`src/models/train_runner.py`).

Inspect the lock:

```bash
cat models/.training.lock
```

Example payload:

```json
{"pid": 12345, "started_at": "2026-07-13T10:00:00+00:00"}
```

### Stale lock

If the PID is no longer running, the next `POST /v1/train` call clears the lock automatically (`clear_stale_lock()` in `src/models/train_runner.py`).

Manual removal is safe only after confirming the PID is dead:

```bash
# Linux/macOS — replace 12345 with the lock PID
kill -0 12345 && echo "still running" || rm models/.training.lock

# Windows PowerShell
Get-Process -Id 12345 -ErrorAction SilentlyContinue; if (-not $?) { Remove-Item models\.training.lock }
```

### Logs and API status

Application logs are written under `logs/` (see `configs/config.yaml` → `paths.logs_dir`).

Authenticated health exposes training state without internal error messages:

```bash
curl -s http://localhost:8000/health -H "X-API-Key: $CHURN_API_KEY" | jq .training_status
```

Fields returned: `status`, `phase`, `last_trained_at` (sanitized in `app/api/fastapi_app.py`).

Typical `status` values: `idle`, `running`, `completed`, `failed`.

### Re-trigger training

Requires an **admin** API key (`CHURN_API_KEY` or a key issued with `--scopes admin`):

```bash
curl -X POST http://localhost:8000/v1/train -H "X-API-Key: $CHURN_API_KEY"
```

| HTTP code | Meaning |
|-----------|---------|
| `200` | Training subprocess started |
| `409` | Training already in progress (in-memory status or active lock) |
| `403` | Key valid but missing `admin` scope |
| `401` | Invalid or missing API key |
| `500` | Subprocess launch failed; check `logs/` and lock file |

Alternative local training (no API):

```bash
python src/models/train_model.py
```

---

## `/v1/champion/status` interpretation

```bash
curl -s http://localhost:8000/v1/champion/status -H "X-API-Key: $CHURN_API_KEY" | jq
```

### Response fields (`ChampionStatusResponse` in `app/api/fastapi_app.py`)

| Field | Description |
|-------|-------------|
| `champion` | Model currently serving predictions (`model_name`, `metrics`, `trained_at`, `promoted_at`) |
| `challenger` | Candidate model held for shadow scoring and promotion comparison |
| `promotion_threshold` | Minimum ROC-AUC improvement required for promotion (from `configs/config.yaml` → `champion_challenger.promotion_threshold`, default `0.005`) |
| `history` | Recent promotion decisions (last 10 entries) |

### History entry fields

| Field | Description |
|-------|-------------|
| `timestamp` | UTC ISO timestamp of the decision |
| `action` | e.g. `initial`, `promote`, `retain`, `manual_rollback` |
| `previous_champion` | Prior champion name (null on first run) |
| `new_champion` | Champion after the decision |
| `champion_metric` | Champion ROC-AUC at decision time |
| `challenger_metric` | Challenger ROC-AUC at decision time |
| `delta` | `challenger_metric - champion_metric` |

### Healthy vs concerning

| Signal | Assessment |
|--------|------------|
| `404` on status | No registry yet — run training |
| `champion` is null | Registry uninitialized |
| `history` empty after training | Unexpected — inspect `models/registry.db` |
| Large negative `delta` in recent `retain` actions | Challenger underperforming; expected if threshold not met |
| Frequent `manual_rollback` | Ops intervention — verify model quality before next train |

Promotion also requires challenger recall ≥ `min_recall_floor` (default `0.70`) per `ModelRegistry` in `src/models/registry.py`.

---

## Manual rollback

### Preconditions

- `models/registry.db` exists
- Both `champion` and `challenger` roles are set (otherwise API returns `400`)

### Command

```bash
curl -X POST http://localhost:8000/v1/champion/rollback \
  -H "X-API-Key: $CHURN_API_KEY"
```

Requires **admin** scope. Response (`RollbackResponse`):

```json
{
  "action": "manual_rollback",
  "champion": "lightgbm",
  "challenger": "logistic_regression",
  "delta": 0.02
}
```

### Verify

1. Re-fetch champion status:

   ```bash
   curl -s http://localhost:8000/v1/champion/status -H "X-API-Key: $CHURN_API_KEY" | jq '.champion.model_name'
   ```

2. Confirm serving model via authenticated health:

   ```bash
   curl -s http://localhost:8000/health -H "X-API-Key: $CHURN_API_KEY" | jq '.served_by_model'
   ```

3. Check dashboard **Model Performance** page or `models/model_metadata.json` for artifact alignment.

Rollback reloads the in-memory predictor when artifacts are present (`app/api/fastapi_app.py` → `champion_rollback`).

---

## Common failures

### SQLite `BEGIN IMMEDIATE` contention

Registry and API keys share `models/registry.db` (`src/models/db.py`). Concurrent writes (training promotion + key issuance + rollback) can block briefly.

**Symptoms:** Slow train/rollback responses, SQLite lock errors in logs.

**Mitigation:** Retry the operation; avoid parallel training triggers; for heavy multi-replica deployments see `docs/ARCHITECTURE.md` (Postgres migration sketch).

### 401 vs 403

| Code | Cause | Log hint |
|------|-------|----------|
| `401` | Missing, revoked, or unknown API key | Invalid key attempts |
| `403` | Valid key without `admin` scope on `/v1/train` or `/v1/champion/rollback` | `"Admin scope required for this endpoint"` |

Predict-scoped keys work on `/v1/predict`, `/v1/predict_batch`, `/v1/champion/status`, `/health`, and `/metrics`.

Issue keys:

```bash
python scripts/manage_keys.py issue inference-bot --scopes predict
python scripts/manage_keys.py issue ops-team --scopes admin
```

### Container troubleshooting

```bash
docker compose -f infra/docker-compose.yml logs api dashboard
docker compose -f infra/docker-compose.yml ps
```

Volume mounts (`infra/docker-compose.yml`):

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Raw and processed data |
| `./models` | `/app/models` | Artifacts, registry, training lock |
| `./logs` | `/app/logs` | Application logs |
| `./mlruns` | `/app/mlruns` | MLflow runs (API only) |

If predictions return `503` (“Model artifacts not found”), confirm `models/best_model.pkl` and related artifacts exist on the mounted volume and that training completed successfully.

### Rate limits

Configured in `configs/config.yaml` → `api.rate_limits`:

- `predict`: 60/minute
- `predict_batch`: 10/minute
- `train`: 1/minute

`429` responses indicate client-side retry/backoff is needed.
