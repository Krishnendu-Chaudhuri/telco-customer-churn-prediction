"""Data loading and schema inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import ColumnSchema, DatasetSchema
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Telco churn dataset."""
    if path is None:
        path = ProjectPaths().raw_data
    path = Path(path)

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %s rows and %s columns", len(df), len(df.columns))
    return df


def _infer_semantic_type(
    column: str,
    series: pd.Series,
    config: dict[str, Any],
) -> str:
    """Infer semantic column type from configuration and values."""
    data_cfg = config["data"]
    if column == data_cfg["id_column"]:
        return "id"
    if column == data_cfg["target_column"]:
        return "target"
    if column in data_cfg["numeric_columns"]:
        return "numeric"
    if column in data_cfg["binary_columns"]:
        return "binary"
    if column in data_cfg["categorical_columns"]:
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def infer_schema(df: pd.DataFrame, config: dict[str, Any] | None = None) -> DatasetSchema:
    """Infer dataset schema including semantic types and summary stats."""
    config = config or get_config()
    columns: list[ColumnSchema] = []

    for column in df.columns:
        series = df[column]
        columns.append(
            ColumnSchema(
                name=column,
                dtype=str(series.dtype),
                semantic_type=_infer_semantic_type(column, series, config),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=series.dropna().astype(str).head(3).tolist(),
            )
        )

    target_col = config["data"]["target_column"]
    target_distribution: dict[str, int] = {}
    if target_col in df.columns:
        target_distribution = df[target_col].value_counts().to_dict()

    return DatasetSchema(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        target_distribution=target_distribution,
    )
