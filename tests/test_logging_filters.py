"""Tests for PII redaction in logs."""

from __future__ import annotations

import logging

from src.utils.logging_filters import RedactingFormatter, redact_payload


def test_redact_payload_masks_customer_fields():
    payload = {
        "customerID": "CUST-123",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.1,
        "TotalCharges": 1000.0,
        "tenure": 12,
    }
    redacted = redact_payload(payload)
    assert redacted["customerID"] == "[REDACTED]"
    assert redacted["PaymentMethod"] == "[REDACTED]"
    assert redacted["MonthlyCharges"] == "[REDACTED]"
    assert redacted["TotalCharges"] == "[REDACTED]"
    assert redacted["tenure"] == "[REDACTED]"


def test_redacting_formatter_masks_log_output():
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="customerID=CUST-999 MonthlyCharges=70.0",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "[REDACTED]" in rendered
    assert "CUST-999" not in rendered
