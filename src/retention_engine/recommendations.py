"""Rule-based retention recommendation engine."""

from __future__ import annotations

from typing import Any

from src.utils.config import get_config


class RetentionEngine:
    """Generate retention strategies based on churn probability."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.thresholds = self.config["thresholds"]
        self.strategies = self.config["retention"]["strategies"]

    def get_risk_level(self, churn_probability: float) -> str:
        """Map churn probability to risk level."""
        if churn_probability >= self.thresholds["critical"]:
            return "Critical"
        if churn_probability >= self.thresholds["high"]:
            return "High"
        if churn_probability >= self.thresholds["medium"]:
            return "Medium"
        return "Low"

    def _strategy_key(self, churn_probability: float) -> str:
        if churn_probability >= self.thresholds["critical"]:
            return "critical"
        if churn_probability >= self.thresholds["high"]:
            return "high"
        if churn_probability >= self.thresholds["medium"]:
            return "medium"
        return "low"

    def recommend(
        self,
        churn_probability: float,
        clv_estimate: float,
    ) -> dict[str, Any]:
        """Generate retention recommendation for a customer."""
        strategy_key = self._strategy_key(churn_probability)
        strategy = self.strategies[strategy_key]
        expected_revenue_saved = (
            churn_probability * clv_estimate * strategy["success_rate"]
        )

        return {
            "churn_probability": float(churn_probability),
            "risk_level": self.get_risk_level(churn_probability),
            "retention_strategy": strategy["name"],
            "retention_cost": float(strategy["cost"]),
            "expected_revenue_saved": float(expected_revenue_saved),
            "clv": float(clv_estimate),
        }
