"""Feature engineering for churn prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Create business features for churn modeling."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        engineered = df.copy()
        engineered = self._add_tenure_group(engineered)
        engineered = self._add_service_count(engineered)
        engineered = self._add_engagement_score(engineered)
        engineered = self._add_payment_risk_score(engineered)
        engineered = self._add_contract_risk(engineered)
        engineered = self._add_revenue_features(engineered)
        engineered = self._add_clv_estimate(engineered)
        engineered = self._add_churn_propensity_indicators(engineered)
        logger.info("Feature engineering completed with %s columns", len(engineered.columns))
        return engineered

    def _add_tenure_group(self, df: pd.DataFrame) -> pd.DataFrame:
        bins = self.config["features"]["tenure_bins"]

        def assign_group(tenure: float) -> str:
            for item in bins:
                if item["min"] <= tenure < item["max"]:
                    return item["label"]
            return bins[-1]["label"]

        df["tenure_group"] = df["tenure"].apply(assign_group)
        return df

    def _is_service_active(self, value: Any) -> int:
        if pd.isna(value):
            return 0
        text = str(value).strip().lower()
        if text in {"yes", "dsl", "fiber optic"}:
            return 1
        return 0

    def _add_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        service_columns = self.config["features"]["service_columns"]
        available = [col for col in service_columns if col in df.columns]
        df["service_count"] = df[available].apply(
            lambda row: sum(self._is_service_active(value) for value in row),
            axis=1,
        )
        return df

    def _add_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.config["features"]["engagement_weights"]
        score = np.zeros(len(df))

        for column, weight in weights.items():
            if column not in df.columns:
                continue
            if column == "InternetService":
                active = df[column].isin(["DSL", "Fiber optic"]).astype(float)
            else:
                active = (df[column] == "Yes").astype(float)
            score += active * weight

        df["engagement_score"] = score
        return df

    def _add_payment_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.config["features"]["payment_risk_weights"]
        score = np.zeros(len(df))

        if "Contract" in df.columns:
            score += (df["Contract"] == "Month-to-month").astype(float) * weights["month_to_month"]
        if "PaymentMethod" in df.columns:
            score += (df["PaymentMethod"] == "Electronic check").astype(float) * weights[
                "electronic_check"
            ]
        if "PaperlessBilling" in df.columns:
            score += (df["PaperlessBilling"] == "Yes").astype(float) * weights["paperless_billing"]

        df["payment_risk_score"] = score
        return df

    def _add_contract_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = self.config["features"]["contract_risk_map"]
        df["contract_risk"] = df["Contract"].map(mapping).fillna(2)
        return df

    def _add_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config["features"]["high_monthly_charge_threshold"]
        df["customer_lifetime_revenue"] = df["MonthlyCharges"] * df["tenure"]
        df["avg_monthly_value"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
        df["revenue_risk"] = (
            (df["tenure"] < 12).astype(float) * 2
            + (df["MonthlyCharges"] > threshold).astype(float) * 2
            + (df["Contract"] == "Month-to-month").astype(float) * 2
        )
        return df

    def _add_clv_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        clv_cfg = self.config["clv"]
        gross_margin = clv_cfg["gross_margin"]
        discount_rate = clv_cfg["discount_rate"]
        lifespan = clv_cfg["avg_lifespan_months"]

        monthly_value = df["avg_monthly_value"]
        tenure_factor = (1 + discount_rate) ** df["tenure"].clip(lower=0)
        base_clv = (monthly_value * gross_margin * lifespan) / tenure_factor
        df["clv_estimate"] = base_clv.fillna(0)
        return df

    def _add_churn_propensity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config["features"]["high_monthly_charge_threshold"]
        low_engagement = self.config["features"]["low_engagement_threshold"]

        df["is_new_customer"] = (df["tenure"] < 12).astype(int)
        df["has_high_charges"] = (df["MonthlyCharges"] > threshold).astype(int)
        df["has_no_contract"] = (df["Contract"] == "Month-to-month").astype(int)
        df["has_low_engagement"] = (df["engagement_score"] < low_engagement).astype(int)
        df["churn_propensity_score"] = (
            df["is_new_customer"]
            + df["has_high_charges"]
            + df["has_no_contract"]
            + df["has_low_engagement"]
            + (df["payment_risk_score"] >= 3).astype(int)
        )
        return df
