"""Testable dashboard business logic (no Streamlit dependencies)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.models.predictor import ChurnPredictor
    from src.models.registry import ModelRegistry

HIGH_RISK_THRESHOLD = 0.7


def compute_executive_kpis(
    df: pd.DataFrame,
    predictor: ChurnPredictor | None = None,
) -> dict[str, float]:
    """Compute executive KPIs from customer data and optional predictor."""
    if "Churn" in df.columns:
        churn_rate = float((df["Churn"] == "Yes").mean())
    else:
        churn_rate = float(df.get("churn_probability", pd.Series([0.0])).mean())

    revenue_at_risk = 0.0
    if "churn_probability" in df.columns and "clv_estimate" in df.columns:
        revenue_at_risk = float((df["churn_probability"] * df["clv_estimate"]).sum())
    elif predictor is not None:
        probs = predictor.predict_proba(df)
        engineered = predictor.pipeline.feature_engineer.transform(
            predictor.pipeline.cleaner.transform(df)
        )
        revenue_at_risk = float((probs * engineered["clv_estimate"]).sum())

    if "churn_probability" in df.columns:
        high_risk_count = int((df["churn_probability"] >= HIGH_RISK_THRESHOLD).sum())
    else:
        high_risk_count = 0

    avg_clv = float(df["clv_estimate"].mean()) if "clv_estimate" in df.columns else 0.0

    return {
        "churn_rate": churn_rate,
        "revenue_at_risk": revenue_at_risk,
        "high_risk_count": high_risk_count,
        "avg_clv": avg_clv,
    }


def segmentation_has_data(df: pd.DataFrame) -> bool:
    """Return True when segmentation columns are present for visualization."""
    return "segment" in df.columns and not df.empty


def build_segment_clv_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate CLV by segment when segmentation and CLV columns exist."""
    if not segmentation_has_data(df) or "clv_estimate" not in df.columns:
        return None
    return df.groupby("segment", as_index=False)["clv_estimate"].sum()


def format_champion_panel(registry_obj: ModelRegistry) -> dict[str, Any]:
    """Assemble champion/challenger panel data from a registry object."""
    champion = registry_obj.champion or {}
    challenger = registry_obj.challenger or {}

    metric_rows: list[dict[str, Any]] = []
    for role_name, role_data in (("Champion", champion), ("Challenger", challenger)):
        metrics = role_data.get("metrics") or {}
        metric_rows.append(
            {
                "role": role_name,
                "model": role_data.get("model_name"),
                **metrics,
            }
        )

    return {
        "champion_model": champion.get("model_name"),
        "challenger_model": challenger.get("model_name"),
        "promotion_threshold": registry_obj.promotion_threshold,
        "metric_rows": metric_rows,
        "history": registry_obj.history[-10:],
    }
