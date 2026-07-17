# Telco Customer Churn & Retention Engine

![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)

Production-grade machine learning system for predicting telecom customer churn, estimating customer lifetime value (CLV), and recommending retention actions.

## Overview

End-to-end data science platform covering data validation, feature engineering, model training, champion/challenger governance, REST API serving, and an executive Streamlit dashboard. Built on the IBM Telco Customer Churn dataset (~7,000 customers).

## Problem Statement

Telecom providers lose revenue when customers churn. This project identifies at-risk customers early, quantifies revenue exposure, segments the base for targeted campaigns, and recommends cost-effective retention strategies.

## Features

- Churn probability scoring with risk tiers (critical / high / medium / low)
- CLV estimation and revenue-at-risk KPIs
- Rule-based retention recommendations with expected savings
- KMeans customer segmentation
- SHAP explainability (global and per-customer)
- Champion/challenger model registry with promotion guardrails
- Versioned FastAPI (`/v1/*`) with scoped API keys
- Streamlit executive dashboard
- MLflow experiment tracking
- Docker Compose deployment

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| ML | scikit-learn, LightGBM, imbalanced-learn, SHAP |
| API | FastAPI, Uvicorn, Pydantic, SlowAPI |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, PyArrow (Parquet) |
| Tracking | MLflow |
| Persistence | SQLite (registry + API keys), Joblib artifacts |
| Ops | Docker, Prometheus, GitHub Actions, Ruff, pytest |

## Architecture

```
Raw CSV → src/data → src/pipelines → src/features → src/models/train
                                                         ↓
                                              models/ artifacts + registry.db
                                                         ↓
                                    app/api (FastAPI) + app/dashboard (Streamlit)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for scaling decisions and [docs/RUNBOOK.md](docs/RUNBOOK.md) for incident response.

## Project Structure

```
.
├── main.py                 # CLI: train | api | dashboard
├── app/
│   ├── api/                # FastAPI service (versioned /v1 routes)
│   └── dashboard/          # Streamlit UI + testable logic
├── src/
│   ├── data/               # Data loading and schema
│   ├── features/           # Feature engineering
│   ├── pipelines/          # Cleaning, encoding, scaling
│   ├── models/             # Training, inference, registry
│   ├── explainability/     # SHAP explanations
│   ├── retention_engine/   # Segmentation and recommendations
│   └── utils/              # Config, logging, API keys
├── configs/config.yaml     # Central configuration (paths, hyperparams, thresholds)
├── data/raw/               # Source dataset
├── docs/                   # Architecture and runbook
├── notebooks/              # Optional EDA notebook
├── scripts/                # API key management, dependency audit
├── tests/                  # Pytest suite (75% coverage gate)
├── assets/                 # Static assets placeholder
├── infra/                  # Dockerfile, docker-compose, .dockerignore
├── requirements.txt        # Python dependencies (runtime + dev/CI)
├── .env.example            # Environment variable template
```

## Installation

```bash
git clone <repo-url>
cd telco-churn-retention
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set CHURN_API_KEY
```

## Configuration

- **YAML:** [`configs/config.yaml`](configs/config.yaml) — paths, model hyperparameters, retention thresholds, champion/challenger rules
- **Environment:** copy [`.env.example`](.env.example)

| Variable | Description | Default |
|----------|-------------|---------|
| `CHURN_API_KEY` | Admin API key | (required) |
| `CHURN_CORS_ORIGINS` | Comma-separated CORS origins | deny-all |
| `CHURN_DEBUG` | Expose exception details in 500 responses | `false` |

## Usage

### Unified CLI

```bash
python main.py train
python main.py api --reload
python main.py dashboard
```

### Direct commands

```bash
python src/models/train_model.py
uvicorn app.api.fastapi_app:app --reload
```

**Important:** `cd` into this repository first, or prefer `python main.py dashboard` (uses absolute paths). Running `streamlit run app/dashboard/streamlit_app.py` from another project's folder will launch that project's dashboard.

```bash
python main.py dashboard
# or, from this repo root only:
streamlit run app/dashboard/streamlit_app.py --server.port 8502
```

## Training Pipeline

1. Validate raw data (`src/pipelines/validator.py`)
2. Clean, encode, scale, engineer features (`src/pipelines`, `src/features`)
3. Train Logistic Regression and LightGBM with hyperparameter tuning
4. Apply SMOTE on the training split (configurable)
5. Evaluate via champion/challenger promotion (ROC-AUC + recall floor)
6. Persist artifacts to `models/`, processed parquet to `data/processed/`
7. Generate evaluation plots and SHAP summary; log to MLflow (`./mlruns`)

## Inference

- **Single:** `POST /v1/predict` with customer JSON body
- **Batch:** `POST /v1/predict_batch`
- **Shadow scoring:** `?shadow=true` returns challenger prediction alongside champion

## API & UI

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | Optional key for extended status | Service health |
| `/v1/predict` | POST | API key | Single prediction |
| `/v1/predict_batch` | POST | API key | Batch predictions |
| `/v1/train` | POST | Admin key | Background training |
| `/v1/champion/status` | GET | API key | Registry status |
| `/v1/champion/rollback` | POST | Admin key | Manual rollback |
| `/metrics` | GET | API key | Prometheus metrics |

**Dashboard pages:** Executive KPIs, Customer Lookup, Segmentation, Model Performance, Explainability.

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features (demographics, services, contract, charges). Stored at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Model

- **Candidates:** Logistic Regression, LightGBM (tuned via randomized search)
- **Selection:** Champion/challenger framework; promotion when challenger beats champion ROC-AUC by ≥ 0.005 with recall ≥ 0.70
- **Serving:** Best model artifacts loaded by `ChurnPredictor`

## Evaluation Metrics

ROC-AUC (primary), recall, precision, F1, accuracy. Holdout evaluation artifacts saved under `models/evaluation/`.

## Results

Typical holdout ROC-AUC ~0.83–0.84 for the champion model. Exact metrics are written to `models/model_metadata.json` after each training run.

## Testing

```bash
pytest tests/ -v -m "not slow"
ruff check .
make audit    # dependency vulnerability scan
```

Coverage gate: **75%** on `src` + `app`.

## Docker Deployment

```bash
docker compose -f infra/docker-compose.yml up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8502

Volumes: `./data`, `./models`, `./logs`, `./mlruns` (mounted from project root)

## Security

Scoped API keys via `scripts/manage_keys.py`:

```bash
python scripts/manage_keys.py issue inference-bot --scopes predict
python scripts/manage_keys.py issue ops-team --scopes admin
```

Keys hashed in `models/registry.db`. See [docs/RUNBOOK.md](docs/RUNBOOK.md) for 401 vs 403 troubleshooting.

## Future Improvements

- PostgreSQL registry backend for multi-replica deployments
- Shared remote MLflow tracking server
- Scheduled batch feature pipeline at higher data volumes

Documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Python 3.11+
- See [`requirements.txt`](requirements.txt) for direct dependencies (runtime and dev tools)

