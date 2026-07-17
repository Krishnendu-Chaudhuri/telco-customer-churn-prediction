"""Feature encoding utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """Encode binary and categorical features for modeling."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.binary_maps = {
            "gender": {"Female": 0, "Male": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "PaperlessBilling": {"No": 0, "Yes": 1},
        }
        self.one_hot_columns: list[str] = []
        self.category_maps_: dict[str, list[str]] = {}
        self.feature_columns_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Learn encoding schema from training data."""
        encoded = self._encode_binary(df.copy())
        encoded = self._encode_one_hot(encoded, fit_mode=True)
        self.feature_columns_ = self._select_model_columns(encoded)
        logger.info("Encoder fitted with %s features", len(self.feature_columns_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted schema."""
        encoded = self._encode_binary(df.copy())
        encoded = self._encode_one_hot(encoded, fit_mode=False)
        for column in self.feature_columns_:
            if column not in encoded.columns:
                encoded[column] = 0
        return encoded[self.feature_columns_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _select_model_columns(self, encoded: pd.DataFrame) -> list[str]:
        excluded = self.config["data"]["drop_columns"] + [self.config["data"]["target_column"]]
        return [col for col in encoded.columns if col not in excluded]

    def _encode_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, mapping in self.binary_maps.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
        return df

    def _encode_one_hot(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        data_cfg = self.config["data"]
        categorical_cols = [
            col
            for col in data_cfg["categorical_columns"] + ["tenure_group"]
            if col in df.columns
        ]
        self.one_hot_columns = categorical_cols

        encoded = df.copy()
        for column in categorical_cols:
            if fit_mode:
                categories = sorted(encoded[column].dropna().astype(str).unique().tolist())
                self.category_maps_[column] = categories
            else:
                categories = self.category_maps_.get(column, [])

            for category in categories[1:]:
                dummy_name = f"{column}_{category}"
                encoded[dummy_name] = (encoded[column].astype(str) == category).astype(int)

        encoded = encoded.drop(columns=categorical_cols, errors="ignore")
        return encoded
