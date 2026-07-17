"""Customer segmentation using KMeans clustering."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerSegmentation:
    """Cluster customers into business segments."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_config()
        self.segment_cfg = self.config["segmentation"]
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.segment_cfg["n_clusters"],
            random_state=self.config["training"]["random_state"],
            n_init=10,
        )
        self.segment_mapping: dict[int, str] = {}
        self.feature_columns_: list[str] = []

    def fit_predict(
        self,
        df: pd.DataFrame,
        churn_probabilities: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit clustering model and assign segment labels."""
        features = self._prepare_features(df, churn_probabilities)
        self.feature_columns_ = features.columns.tolist()
        scaled = self.scaler.fit_transform(features)
        cluster_ids = self.kmeans.fit_predict(scaled)
        self.segment_mapping = self._map_clusters_to_labels(
            features, cluster_ids, churn_probabilities
        )

        result = df.copy()
        result["cluster_id"] = cluster_ids
        result["segment"] = [self.segment_mapping[cid] for cid in cluster_ids]
        if churn_probabilities is not None:
            result["churn_probability"] = churn_probabilities.values
        return result

    def _prepare_features(
        self,
        df: pd.DataFrame,
        churn_probabilities: pd.Series | None,
    ) -> pd.DataFrame:
        feature_cols = self.segment_cfg["features"]
        available = [col for col in feature_cols if col in df.columns]
        features = df[available].copy()

        if churn_probabilities is not None:
            features["churn_probability"] = churn_probabilities.values
        elif "churn_probability" not in features.columns:
            features["churn_probability"] = 0.0

        return features.fillna(0)

    def _build_cluster_profiles(
        self,
        features: pd.DataFrame,
        cluster_ids: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Build centroid profiles for each cluster."""
        profiles: list[dict[str, Any]] = []

        for cluster_id in sorted(set(cluster_ids)):
            mask = cluster_ids == cluster_id
            profile = {
                "cluster_id": int(cluster_id),
                "tenure": float(features.loc[mask, "tenure"].mean())
                if "tenure" in features
                else 0.0,
                "monthly_charges": float(features.loc[mask, "MonthlyCharges"].mean())
                if "MonthlyCharges" in features
                else 0.0,
                "clv_estimate": float(features.loc[mask, "clv_estimate"].mean())
                if "clv_estimate" in features
                else 0.0,
                "engagement_score": float(features.loc[mask, "engagement_score"].mean())
                if "engagement_score" in features
                else 0.0,
                "churn_probability": float(features.loc[mask, "churn_probability"].mean())
                if "churn_probability" in features
                else 0.0,
            }
            profiles.append(profile)

        return profiles

    def _label_score(
        self, profile: dict[str, Any], label: str, profiles: list[dict[str, Any]]
    ) -> float:
        """Score how well a cluster profile matches a business label."""
        clv_values = [p["clv_estimate"] for p in profiles]
        charge_values = [p["monthly_charges"] for p in profiles]
        clv_threshold = float(np.percentile(clv_values, 60)) if clv_values else 0.0
        charge_p50 = float(np.percentile(charge_values, 50)) if charge_values else 0.0
        charge_p40 = float(np.percentile(charge_values, 40)) if charge_values else 0.0

        if label == "High Value Loyal":
            score = 0.0
            if profile["clv_estimate"] >= clv_threshold:
                score += 2.0
            if profile["churn_probability"] <= 0.35:
                score += 2.0
            if profile["tenure"] >= 24:
                score += 1.5
            return score

        if label == "High Risk High Revenue":
            score = 0.0
            if profile["churn_probability"] >= 0.55:
                score += 2.5
            if profile["monthly_charges"] >= charge_p50:
                score += 2.0
            return score

        if label == "Low Revenue Churners":
            score = 0.0
            if profile["churn_probability"] >= 0.45:
                score += 2.0
            if profile["monthly_charges"] <= charge_p40:
                score += 2.0
            return score

        if label == "New At Risk":
            score = 0.0
            if profile["tenure"] < 12:
                score += 3.0
            if profile["churn_probability"] >= 0.4:
                score += 1.5
            return score

        # Stable Customers fallback score
        return 1.0 + max(0.0, 0.5 - profile["churn_probability"])

    def _map_clusters_to_labels(
        self,
        features: pd.DataFrame,
        cluster_ids: np.ndarray,
        churn_probabilities: pd.Series | None,
    ) -> dict[int, str]:
        """Map KMeans clusters to business segment labels with unique 1:1 assignment."""
        labels = list(self.segment_cfg["segment_labels"])
        profiles = self._build_cluster_profiles(features, cluster_ids)
        if not profiles:
            return {}

        mapping: dict[int, str] = {}
        assigned_labels: set[str] = set()
        priority = [
            "High Value Loyal",
            "High Risk High Revenue",
            "Low Revenue Churners",
            "New At Risk",
            "Stable Customers",
        ]

        remaining_clusters = {profile["cluster_id"] for profile in profiles}

        for label in priority:
            if label not in labels or label in assigned_labels:
                continue
            best_profile = None
            best_score = float("-inf")
            for profile in profiles:
                cluster_id = profile["cluster_id"]
                if cluster_id not in remaining_clusters:
                    continue
                score = self._label_score(profile, label, profiles)
                if score > best_score:
                    best_score = score
                    best_profile = profile
            if best_profile is not None and best_score > 0:
                mapping[int(best_profile["cluster_id"])] = label
                assigned_labels.add(label)
                remaining_clusters.remove(int(best_profile["cluster_id"]))

        leftover_labels = [label for label in labels if label not in assigned_labels]
        for cluster_id in sorted(remaining_clusters):
            label = leftover_labels.pop(0) if leftover_labels else "Stable Customers"
            mapping[int(cluster_id)] = label
            assigned_labels.add(label)

        for cluster_id in range(self.segment_cfg["n_clusters"]):
            if cluster_id not in mapping:
                label = leftover_labels.pop(0) if leftover_labels else "Stable Customers"
                mapping[int(cluster_id)] = label

        return mapping

    def predict(
        self,
        df: pd.DataFrame,
        churn_probabilities: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Assign segments using a fitted clustering model."""
        features = self._prepare_features(df, churn_probabilities)
        for column in self.feature_columns_:
            if column not in features.columns:
                features[column] = 0.0
        features = features[self.feature_columns_]
        scaled = self.scaler.transform(features)
        cluster_ids = self.kmeans.predict(scaled)

        result = df.copy()
        result["cluster_id"] = cluster_ids
        result["segment"] = [
            self.segment_mapping.get(int(cid), "Stable Customers") for cid in cluster_ids
        ]
        if churn_probabilities is not None:
            result["churn_probability"] = churn_probabilities.values
        return result

    def save(self, model_path: str, mapping_path: str) -> None:
        """Persist segmentation artifacts."""
        joblib.dump(
            {
                "scaler": self.scaler,
                "kmeans": self.kmeans,
                "feature_columns": self.feature_columns_,
            },
            model_path,
        )
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(self.segment_mapping, handle, indent=2)
        logger.info("Saved segmentation artifacts")

    @classmethod
    def load(
        cls, model_path: str, mapping_path: str, config: dict[str, Any] | None = None
    ) -> "CustomerSegmentation":
        """Load segmentation artifacts."""
        instance = cls(config=config)
        payload = joblib.load(model_path)
        instance.scaler = payload["scaler"]
        instance.kmeans = payload["kmeans"]
        instance.feature_columns_ = payload["feature_columns"]
        with open(mapping_path, "r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        instance.segment_mapping = {int(k): v for k, v in raw_mapping.items()}
        return instance
