"""Inference utilities for churn prediction."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.registry import ModelRegistry
from src.pipelines.pipeline import PreprocessingPipeline
from src.retention_engine.recommendations import RetentionEngine
from src.retention_engine.segmentation import CustomerSegmentation
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


class ChurnPredictor:
    """Load artifacts and serve churn predictions."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()

        self.paths = ProjectPaths(self.config)

        self.pipeline: PreprocessingPipeline | None = None

        self.model: Any = None

        self.challenger_model: Any = None

        self.champion_name: str | None = None

        self.challenger_name: str | None = None

        self.retention_engine = RetentionEngine(self.config)

        self.segmentation: CustomerSegmentation | None = None

        self.shap_explainer: Any | None = None

        self.metadata: dict[str, Any] = {}

        self.feature_columns: list[str] = []

    @property
    def is_ready(self) -> bool:
        """Return True when required artifacts are available."""

        if not self.paths.feature_pipeline.exists() or not self.paths.model_metadata.exists():
            return False

        if self.paths.registry_db.exists():
            try:
                registry = ModelRegistry.load(
                    config=self.config,
                    registry_path=self.paths.registry_db,
                )

                champion_name = registry.get_current_champion_name()

                if champion_name and self.paths.model_artifact(champion_name).exists():
                    return True

            except Exception:
                return False

        return self.paths.best_model.exists()

    def load(self) -> "ChurnPredictor":
        """Load model and preprocessing artifacts."""

        if not self.is_ready:
            raise FileNotFoundError(
                "Model artifacts not found. Run `python src/models/train_model.py` first."
            )

        self.pipeline = PreprocessingPipeline.load(str(self.paths.feature_pipeline))

        with self.paths.model_metadata.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.feature_columns = self.metadata.get("feature_columns", self.pipeline.feature_columns_)

        self.champion_name, self.challenger_name, champion_path = self._resolve_serving_paths()

        self.model = joblib.load(champion_path)

        self.challenger_model = None

        if self.paths.kmeans_model.exists() and self.paths.segment_mapping.exists():
            self.segmentation = CustomerSegmentation.load(
                str(self.paths.kmeans_model),
                str(self.paths.segment_mapping),
                self.config,
            )

        try:
            from src.data.data_loader import load_raw_data
            from src.explainability.shap_explainer import ShapExplainer

            background, _, _ = self.pipeline.transform(
                load_raw_data().head(100),
                include_target=False,
            )

            self.shap_explainer = ShapExplainer(
                self.model,
                self.feature_columns,
                background_data=background,
            )

        except Exception as exc:
            logger.warning("SHAP explainer could not be initialized: %s", exc)

            self.shap_explainer = None

        logger.info("Loaded churn predictor with champion model: %s", self.champion_name)

        return self

    def _resolve_serving_paths(self) -> tuple[str, str | None, Any]:
        """Resolve champion/challenger names and the champion artifact path."""

        if self.paths.registry_db.exists():
            registry = ModelRegistry.load(
                config=self.config,
                registry_path=self.paths.registry_db,
            )

            champion_name = registry.get_current_champion_name()

            challenger_name = registry.get_current_challenger_name()

            if champion_name:
                champion_path = self.paths.model_artifact(champion_name)

                if champion_path.exists():
                    return champion_name, challenger_name, champion_path

        champion_name = self.metadata.get("best_model_name", "best_model")

        return champion_name, None, self.paths.best_model

    def _get_challenger_model(self) -> Any:
        """Lazy-load the challenger model artifact."""

        if self.challenger_model is not None:
            return self.challenger_model

        if not self.challenger_name:
            raise ValueError("Challenger model is not configured")

        challenger_path = self.paths.model_artifact(self.challenger_name)

        if not challenger_path.exists():
            raise FileNotFoundError(f"Challenger artifact not found: {challenger_path}")

        self.challenger_model = joblib.load(challenger_path)

        return self.challenger_model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities using the champion model."""

        if self.pipeline is None or self.model is None:
            self.load()

        features, _, _ = self.pipeline.transform(df, include_target=False)

        return self.model.predict_proba(features)[:, 1]

    def predict_batch(
        self,
        df: pd.DataFrame,
        include_explanation: bool = False,
        shadow: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate predictions for a batch of customers."""

        if self.pipeline is None or self.model is None:
            self.load()

        features, _, ids = self.pipeline.transform(df, include_target=False)

        probabilities = self.model.predict_proba(features)[:, 1]

        engineered = self.pipeline.feature_engineer.transform(self.pipeline.cleaner.transform(df))

        challenger_probabilities = None

        if shadow and self.challenger_name:
            challenger_model = self._get_challenger_model()

            challenger_probabilities = challenger_model.predict_proba(features)[:, 1]

        results: list[dict[str, Any]] = []

        for idx, probability in enumerate(probabilities):
            clv = float(engineered.iloc[idx]["clv_estimate"])

            recommendation = self.retention_engine.recommend(float(probability), clv)

            record = {
                "customerID": ids.iloc[idx] if ids is not None else None,
                "served_by_model": self.champion_name,
                **recommendation,
            }

            if include_explanation and self.shap_explainer is not None:
                explanation = self.shap_explainer.explain_instance(features.iloc[[idx]])

                record["explanation"] = explanation["explanation"]

                record["top_contributors"] = explanation["top_contributors"]

            if shadow and challenger_probabilities is not None and self.challenger_name:
                challenger_probability = float(challenger_probabilities[idx])

                record["challenger_prediction"] = {
                    "model_name": self.challenger_name,
                    "churn_probability": challenger_probability,
                    "prediction": int(challenger_probability >= 0.5),
                }

            results.append(record)

        return results

    def predict_single(
        self,
        customer_data: dict[str, Any],
        include_explanation: bool = False,
        shadow: bool = False,
    ) -> dict[str, Any]:
        """Predict churn for a single customer record."""

        df = pd.DataFrame([customer_data])

        return self.predict_batch(
            df,
            include_explanation=include_explanation,
            shadow=shadow,
        )[0]

    def get_segment(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Assign customer segments if segmentation artifacts exist."""

        if self.segmentation is None:
            if self.paths.kmeans_model.exists() and self.paths.segment_mapping.exists():
                self.segmentation = CustomerSegmentation.load(
                    str(self.paths.kmeans_model),
                    str(self.paths.segment_mapping),
                    self.config,
                )

            else:
                return None

        engineered = self.pipeline.feature_engineer.transform(self.pipeline.cleaner.transform(df))

        probabilities = pd.Series(self.predict_proba(df))

        return self.segmentation.predict(engineered, probabilities)
