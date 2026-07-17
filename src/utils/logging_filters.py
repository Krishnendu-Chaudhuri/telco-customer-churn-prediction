"""PII redaction utilities for logging."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from src.utils.config import get_config

REDACTED = "[REDACTED]"

DEFAULT_PII_FIELDS = {
    "customerID",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}


def get_pii_fields() -> set[str]:
    """Return configured and default PII field names."""
    config = get_config()
    data_cfg = config.get("data", {})
    fields = set(DEFAULT_PII_FIELDS)
    fields.add(data_cfg.get("id_column", "customerID"))
    fields.update(data_cfg.get("numeric_columns", []))
    fields.update(data_cfg.get("binary_columns", []))
    fields.update(data_cfg.get("categorical_columns", []))
    return fields


def redact_payload(data: Any, pii_fields: set[str] | None = None) -> Any:
    """Recursively redact sensitive fields from dictionaries and strings."""
    pii_fields = pii_fields or get_pii_fields()

    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if key in pii_fields:
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_payload(value, pii_fields)
        return redacted

    if isinstance(data, list):
        return [redact_payload(item, pii_fields) for item in data]

    if isinstance(data, str):
        redacted_text = data
        for field in sorted(pii_fields, key=len, reverse=True):
            pattern = rf"('{field}'|\"{field}\")\s*:\s*('[^']*'|\"[^\"]*\"|\d+\.?\d*)"
            redacted_text = re.sub(pattern, rf"\1: '{REDACTED}'", redacted_text)
            pattern2 = rf"({field}=)([^\s,}}]+)"
            redacted_text = re.sub(pattern2, rf"\1{REDACTED}", redacted_text)
        return redacted_text

    return data


class PIIRedactionFilter(logging.Filter):
    """Logging filter that redacts PII patterns from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        pii_fields = get_pii_fields()
        if isinstance(record.msg, str):
            record.msg = redact_payload(record.msg, pii_fields)
        elif isinstance(record.msg, dict):
            record.msg = redact_payload(deepcopy(record.msg), pii_fields)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_payload(deepcopy(record.args), pii_fields)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_payload(arg, pii_fields) if isinstance(arg, (str, dict, list)) else arg
                    for arg in record.args
                )
        return True


class RedactingFormatter(logging.Formatter):
    """Formatter that applies PII redaction to the final rendered message."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_payload(rendered, get_pii_fields())
