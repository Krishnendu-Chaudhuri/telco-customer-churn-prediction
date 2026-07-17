"""Backward-compatible import path for serialized artifacts."""

from src.data.data_loader import infer_schema, load_raw_data

__all__ = ["load_raw_data", "infer_schema"]
