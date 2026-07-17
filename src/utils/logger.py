"""Structured logging utilities."""

from __future__ import annotations

import logging
import sys

from src.utils.logging_filters import PIIRedactionFilter, RedactingFormatter
from src.utils.paths import get_project_root


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create or return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = RedactingFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pii_filter = PIIRedactionFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(pii_filter)
    logger.addHandler(console_handler)

    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_name = log_file or f"{name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(logs_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(pii_filter)
    logger.addHandler(file_handler)

    return logger
