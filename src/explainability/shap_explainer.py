"""SHAP-based model explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """Generate global and local SHAP explanations."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pd.DataFrame | np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = self._build_explainer(model, background_data)
        self.expected_value = getattr(self.explainer, "expected_value", 0.0)

    def _build_explainer(
        self,
        model: Any,
        background_data: pd.DataFrame | np.ndarray | None,
    ) -> Any:
        model_name = model.__class__.__name__.lower()
        if any(key in model_name for key in ["xgb", "lgbm", "catboost", "forest", "gradient"]):
            return shap.TreeExplainer(model)

        if background_data is not None:
            if hasattr(model, "coef_"):
                masker = shap.maskers.Independent(background_data)
                return shap.LinearExplainer(model, masker)
            return shap.Explainer(model.predict_proba, background_data)

        if hasattr(model, "predict_proba"):
            return shap.Explainer(model.predict_proba)
        return shap.Explainer(model)

    def explain_instance(self, features: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Explain a single customer prediction."""
        if isinstance(features, pd.DataFrame):
            array = features.values
            columns = features.columns.tolist()
        else:
            array = np.asarray(features)
            columns = self.feature_names

        shap_values = self.explainer.shap_values(array)
        shap_array = self._normalize_shap_values(shap_values, len(array))
        instance_values = shap_array[0]
        contributors = sorted(
            zip(columns, instance_values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]

        summary = self._build_summary(contributors)
        return {
            "top_contributors": [
                {"feature": name, "shap_value": float(value)} for name, value in contributors
            ],
            "explanation": summary,
        }

    def _build_summary(self, contributors: list[tuple[str, float]]) -> str:
        reasons = []
        for feature, value in contributors[:3]:
            direction = "increases" if value > 0 else "decreases"
            clean_name = feature.replace("_", " ")
            reasons.append(f"{clean_name} {direction} churn risk")
        if not reasons:
            return "Insufficient data to explain churn risk."
        return "Why this customer may churn: " + "; ".join(reasons) + "."

    def _normalize_shap_values(
        self,
        shap_values: Any,
        sample_size: int,
    ) -> np.ndarray:
        """Normalize SHAP outputs to a 2D array."""
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
        if shap_array.ndim == 1:
            shap_array = shap_array.reshape(1, -1)
        return shap_array[:sample_size]

    def save_global_summary(
        self,
        x_sample: pd.DataFrame,
        output_path: Path,
        max_samples: int = 500,
    ) -> None:
        """Save global SHAP summary plot."""
        sample = x_sample.head(max_samples)
        shap_values = self.explainer.shap_values(sample)
        shap_array = self._normalize_shap_values(shap_values, len(sample))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_array,
                sample,
                show=False,
                max_display=15,
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            logger.warning("SHAP beeswarm plot failed, using bar chart fallback: %s", exc)
            mean_abs = np.abs(shap_array).mean(axis=0)
            importance = (
                pd.DataFrame({"feature": sample.columns, "importance": mean_abs})
                .sort_values("importance", ascending=True)
                .tail(15)
            )
            plt.figure(figsize=(10, 6))
            plt.barh(importance["feature"], importance["importance"])
            plt.title("Mean |SHAP| Feature Importance")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

        logger.info("Saved SHAP summary plot to %s", output_path)
