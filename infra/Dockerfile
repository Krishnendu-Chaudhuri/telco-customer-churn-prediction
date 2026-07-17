FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY .streamlit/ .streamlit/

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p models/evaluation logs data/processed mlruns models/.staging \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8502

CMD ["uvicorn", "app.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
