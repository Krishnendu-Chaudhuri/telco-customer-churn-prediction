# Architecture Decisions

Scaling and platform choices for the Telco Churn & Retention Engine. This document records **decision context only** — no migration code is implemented today.

## Current system overview

```mermaid
flowchart LR
    CSV[data/raw CSV] --> Loader[data_loader.py]
    Loader --> Pipeline[src/pipelines/pipeline.py]
    Pipeline --> Features[src/features/features.py]
    Features --> Train[train_model.py]
    Train --> Artifacts[models/ artifacts]
    Train --> Registry[(models/registry.db)]
    Artifacts --> API[FastAPI /v1]
    Artifacts --> Dashboard[Streamlit dashboard]
    Train --> MLflow[(./mlruns local)]
```

---

## Registry: SQLite → PostgreSQL

### Current state

- **Database:** SQLite at `models/registry.db`
- **Schema:** `src/models/db.py` — tables `model_roles`, `promotion_history`, `registry_meta`, `api_keys`
- **Access:** `src/models/registry.py` (champion/challenger), `src/utils/api_keys.py` (per-client keys)
- **Transactions:** `BEGIN IMMEDIATE` via `transaction()` context manager for atomic promotion and key writes

### Trigger for migration

Move to PostgreSQL when deploying **multiple API replicas** that write concurrently. SQLite serializes writers; `BEGIN IMMEDIATE` contention under parallel training triggers, rollbacks, and key issuance becomes a bottleneck.

Single-node Docker Compose and one trainer process are fine with SQLite today.

### Future approach (sketch)

1. **Config flag** in `configs/config.yaml`:

   ```yaml
   registry_backend: sqlite   # or postgres
   postgres:
     dsn: postgresql://user:pass@host:5432/churn
   ```

2. **Driver:** `psycopg2` (or `psycopg`) with parameterized SQL — preserve existing query style from `registry.py` and `api_keys.py`.

3. **Abstraction:** Thin backend module (`src/models/db_backend.py`) implementing `get_connection()`, `init_schema()`, `transaction()` for both backends.

4. **Migration script:** One-time export from SQLite (`model_roles`, `promotion_history`, `registry_meta`, `api_keys`) → Postgres `COPY` or `INSERT … SELECT`.

5. **Rollout:** Blue/green — point new replicas at Postgres; keep SQLite for local dev default.

### Tradeoffs

| SQLite | PostgreSQL |
|--------|------------|
| Zero ops overhead, single file | Requires managed DB or container |
| Fine for single writer | Row-level locking, multi-replica safe |
| Bundled with app | Network dependency, connection pooling needed |

---

## MLflow: local → shared tracking server

### Current state

- **Config:** `configs/config.yaml` → `mlflow.tracking_uri: "./mlruns"`
- **Usage:** Training logs experiments via `_log_to_mlflow()` in `src/models/train_model.py`
- **Storage:** Local `./mlruns` directory (mounted in Docker Compose for the API service)

### Trigger for migration

Enable a **shared tracking backend** when multiple trainers or CI jobs need a unified experiment history, or when artifact storage must outlive ephemeral training hosts.

### Future approach (sketch)

1. Set remote URI in config:

   ```yaml
   mlflow:
     experiment_name: telco_churn_retention
     tracking_uri: https://mlflow.internal.example.com
   ```

2. Optionally separate **artifact store** (S3/GCS) from tracking server per MLflow deployment docs.

3. Keep `./mlruns` as the default for local development and offline notebooks.

4. No application code changes beyond config if the experiment name and logging calls remain the same.

### Tradeoffs

| Local `./mlruns` | Remote MLflow |
|------------------|---------------|
| No infra | Centralized compare across runs |
| Lost if volume not mounted | Depends on server availability |
| Fast iteration | Better audit trail for production training |

---

## Feature store / batch pipeline

### Current path

| Stage | Module | Output |
|-------|--------|--------|
| Ingestion | `src/data/data_loader.py` | Raw CSV → DataFrame |
| Preprocessing | `src/pipelines/pipeline.py` | Clean, encode, scale |
| Feature engineering | `src/features/features.py` | Tenure groups, CLV (`clv_estimate`), engagement scores, payment/contract risk (per `configs/config.yaml` → `features`) |
| Training | `src/models/train_model.py` | `data/processed/processed_customers.parquet` |
| Serving | `src/models/predictor.py` | On-the-fly transform at inference |
| Dashboard | `app/dashboard/streamlit_app.py` → `load_data()` | Reads parquet or falls back to raw CSV |

Training fits the preprocessing pipeline once, persists artifacts under `models/`, and writes processed parquet for analytics. Inference applies the same pipeline loaded from disk — no separate feature store.

### Scale threshold

The current batch-in-memory + parquet pattern is appropriate for the Telco dataset scale (~7k rows). Revisit when:

- Row count exceeds **low hundreds of thousands**, or
- Batch feature latency exceeds SLA (e.g. dashboard refresh or batch scoring),

### Candidate approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Scheduled batch parquet/table** (e.g. nightly job → `processed_customers.parquet` or warehouse table) | Simple extension of today; cheap | Stale features between runs; no point-in-time joins |
| **Warehouse tables + dbt/SQL transforms** | Familiar to analytics teams; versioned SQL | Still not real-time; needs orchestration |
| **Dedicated feature store** (Feast, Tecton, etc.) | Online/offline consistency; training/serving skew reduction | Operational complexity; overkill at current volume |

### Recommended sequence

1. Stay on parquet + persisted sklearn pipeline (now).
2. If data grows, add a **scheduled batch job** writing parquet or a warehouse table — dashboard and training read the same snapshot.
3. Consider a feature store only when **online serving** needs sub-minute feature freshness across many models or teams.

---

## API versioning

Business endpoints are mounted under `/v1` (`app/api/fastapi_app.py`). `/health` and `/metrics` remain unversioned. Direct cutover — no legacy redirects (pre-1.0 internal tool).

## Authentication scopes

| Scope | Endpoints |
|-------|-----------|
| `predict` | `/v1/predict`, `/v1/predict_batch`, `/v1/champion/status`, `/health`, `/metrics` |
| `admin` | Above plus `/v1/train`, `/v1/champion/rollback` |

`CHURN_API_KEY` retains full admin access for backward compatibility.
