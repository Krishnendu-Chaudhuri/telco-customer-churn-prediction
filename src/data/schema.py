"""Typed schema definitions for the Telco churn dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnSchema:
    """Schema metadata for a single column."""

    name: str
    dtype: str
    semantic_type: str
    null_count: int
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Schema metadata for the full dataset."""

    row_count: int
    column_count: int
    columns: list[ColumnSchema]
    target_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target_distribution": self.target_distribution,
            "columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                    "semantic_type": col.semantic_type,
                    "null_count": col.null_count,
                    "unique_count": col.unique_count,
                    "sample_values": col.sample_values,
                }
                for col in self.columns
            ],
        }
