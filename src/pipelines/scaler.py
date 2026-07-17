"""Feature scaling utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureScaler:
    """Scale numeric model features using StandardScaler."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.scaler = StandardScaler()
        self.numeric_columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on numeric columns."""
        self.numeric_columns_ = df.select_dtypes(include=["number"]).columns.tolist()
        if self.numeric_columns_:
            self.scaler.fit(df[self.numeric_columns_])
            logger.info("Scaler fitted on %s numeric columns", len(self.numeric_columns_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric columns."""
        if not self.numeric_columns_:
            return df

        scaled = df.copy()
        for column in self.numeric_columns_:
            if column not in scaled.columns:
                scaled[column] = 0.0
        scaled[self.numeric_columns_] = self.scaler.transform(scaled[self.numeric_columns_])
        return scaled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
