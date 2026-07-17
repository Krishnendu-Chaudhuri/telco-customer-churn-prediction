"""Data validation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate schema, data quality, and outliers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run full validation suite and return a report."""
        data_cfg = self.config["data"]
        report: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": self._missing_values(df),
            "duplicates": self._duplicate_report(df, data_cfg["id_column"]),
            "dtype_issues": self._dtype_issues(df),
            "outliers": self._outlier_report(df),
            "target_distribution": {},
            "critical_issues": [],
        }

        target_col = data_cfg["target_column"]
        if target_col in df.columns:
            report["target_distribution"] = df[target_col].value_counts().to_dict()

        if report["duplicates"]["duplicate_ids"] > 0:
            report["critical_issues"].append("Duplicate customer IDs detected")

        for column, count in report["missing_values"].items():
            if count > 0 and column not in {"TotalCharges"}:
                report["critical_issues"].append(f"Missing values in {column}: {count}")

        if report["critical_issues"]:
            logger.warning("Validation critical issues: %s", report["critical_issues"])
            if self.config["validation"].get("fail_on_critical", False):
                raise ValueError(f"Critical validation failures: {report['critical_issues']}")

        logger.info("Validation completed with %s issues", len(report["critical_issues"]))
        return report

    def _missing_values(self, df: pd.DataFrame) -> dict[str, int]:
        missing = df.isna().sum()
        blank_counts = {
            col: int((df[col].astype(str).str.strip() == "").sum())
            for col in df.select_dtypes(include="object").columns
        }
        combined = {col: int(missing.get(col, 0)) for col in df.columns}
        for col, count in blank_counts.items():
            combined[col] = combined.get(col, 0) + count
        return {k: v for k, v in combined.items() if v > 0}

    def _duplicate_report(self, df: pd.DataFrame, id_column: str) -> dict[str, int]:
        duplicate_ids = int(df[id_column].duplicated().sum()) if id_column in df.columns else 0
        duplicate_rows = int(df.duplicated().sum())
        return {
            "duplicate_ids": duplicate_ids,
            "duplicate_rows": duplicate_rows,
        }

    def _dtype_issues(self, df: pd.DataFrame) -> list[str]:
        issues: list[str] = []
        for column in self.config["data"]["numeric_columns"]:
            if column not in df.columns:
                issues.append(f"Missing expected numeric column: {column}")
                continue
            coerced = pd.to_numeric(df[column], errors="coerce")
            invalid = int(coerced.isna().sum() - df[column].isna().sum())
            if invalid > 0:
                issues.append(f"Non-numeric values found in {column}: {invalid}")
        return issues

    def _outlier_report(self, df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        multiplier = self.config["validation"]["iqr_multiplier"]
        outlier_columns = self.config["validation"]["outlier_columns"]
        report: dict[str, dict[str, float | int]] = {}

        for column in outlier_columns:
            if column not in df.columns:
                continue
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            mask = (series < lower) | (series > upper)
            report[column] = {
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(mask.sum()),
            }
        return report
