"""Data cleaning utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean and impute raw customer data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.numeric_medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn imputation statistics from training data."""
        cleaned = self._basic_clean(df.copy())
        for column in self.config["data"]["numeric_columns"]:
            if column in cleaned.columns:
                self.numeric_medians[column] = float(
                    pd.to_numeric(cleaned[column], errors="coerce").median()
                )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations."""
        cleaned = self._basic_clean(df.copy())
        return self._impute_numeric(cleaned)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        data_cfg = self.config["data"]
        id_column = data_cfg["id_column"]

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        for column in data_cfg["numeric_columns"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

        if id_column in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[id_column], keep="first")
            removed = before - len(df)
            if removed:
                logger.info("Removed %s duplicate customer IDs", removed)

        return df

    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, median_value in self.numeric_medians.items():
            if column in df.columns:
                df[column] = df[column].fillna(median_value)
        return df
