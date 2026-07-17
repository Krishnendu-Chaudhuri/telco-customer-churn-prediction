"""End-to-end preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.features.features import FeatureEngineer
from src.pipelines.cleaner import DataCleaner
from src.pipelines.encoder import FeatureEncoder
from src.pipelines.scaler import FeatureScaler
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Orchestrate cleaning, feature engineering, encoding, and scaling."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.cleaner = DataCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.encoder = FeatureEncoder(self.config)
        self.scaler = FeatureScaler(self.config)
        self.feature_columns_: list[str] = []
        self.id_column = self.config["data"]["id_column"]
        self.target_column = self.config["data"]["target_column"]

    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> "PreprocessingPipeline":
        """Fit preprocessing components on training data."""
        cleaned = self.cleaner.fit_transform(df)
        engineered = self.feature_engineer.transform(cleaned)
        encoded = self.encoder.fit_transform(engineered)
        scaled = self.scaler.fit_transform(encoded)
        self.feature_columns_ = scaled.columns.tolist()
        logger.info("Preprocessing pipeline fitted with %s features", len(self.feature_columns_))
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
        """Transform raw data into model-ready features."""
        ids = df[self.id_column] if self.id_column in df.columns else None
        target = None
        if include_target and self.target_column in df.columns:
            target = df[self.target_column].map({"Yes": 1, "No": 0})

        cleaned = self.cleaner.transform(df)
        engineered = self.feature_engineer.transform(cleaned)
        encoded = self.encoder.transform(engineered)
        scaled = self.scaler.transform(encoded)

        for column in self.feature_columns_:
            if column not in scaled.columns:
                scaled[column] = 0.0
        features = scaled[self.feature_columns_]
        return features, target, ids

    def fit_transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, include_target=include_target)

    def save(self, path: str) -> None:
        """Persist preprocessing pipeline."""
        joblib.dump(self, path)
        logger.info("Saved preprocessing pipeline to %s", path)

    @classmethod
    def load(cls, path: str) -> "PreprocessingPipeline":
        """Load preprocessing pipeline from disk."""
        pipeline = joblib.load(path)
        logger.info("Loaded preprocessing pipeline from %s", path)
        return pipeline
