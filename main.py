"""Unified CLI entry point for training, API, and dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_train() -> None:
    from src.models.train_model import train

    train()


def _run_api(host: str, port: int, reload: bool) -> None:
    import uvicorn

    uvicorn.run(
        "app.api.fastapi_app:app",
        host=host,
        port=port,
        reload=reload,
    )


def _run_dashboard() -> None:
    dashboard = PROJECT_ROOT / "app" / "dashboard" / "streamlit_app.py"
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard),
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8502",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Churn & Retention Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run the full training pipeline")

    api_parser = subparsers.add_parser("api", help="Start the FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")

    subparsers.add_parser("dashboard", help="Start the Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "train":
        _run_train()
    elif args.command == "api":
        _run_api(args.host, args.port, args.reload)
    elif args.command == "dashboard":
        _run_dashboard()


if __name__ == "__main__":
    main()
