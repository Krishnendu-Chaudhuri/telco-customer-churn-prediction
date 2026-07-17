"""Model training pipeline entry point."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

_bootstrap_root = Path(__file__).resolve().parents[2]
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from src.utils.paths import ensure_project_imports

PROJECT_ROOT = ensure_project_imports(Path(__file__))


from src.data.data_loader import load_raw_data
from src.models.artifact_publisher import ArtifactPublisher
from src.models.evaluate import (
    build_comparison_table,
    build_model_result,
    compute_metrics,
    save_evaluation_artifacts,
)
from src.models.registry import ModelRegistry
from src.models.tuning import tune_models
from src.pipelines.pipeline import PreprocessingPipeline
from src.pipelines.validator import DataValidator
from src.retention_engine.segmentation import CustomerSegmentation
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


def _validate(raw_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run dataset validation and return the validation report."""
    validator = DataValidator(config)
    return validator.validate(raw_df)


def _engineer_features(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series | None,
    PreprocessingPipeline,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Fit preprocessing and split features into train/test sets."""
    pipeline = PreprocessingPipeline(config)
    features, target, ids = pipeline.fit_transform(raw_df, include_target=True)
    if target is None:
        raise ValueError("Target column not found in dataset")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=target,
    )
    return features, target, ids, pipeline, x_train, x_test, y_train, y_test


def _apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally balance the training set with SMOTE."""
    if config["training"]["smote"]["enabled"]:
        smote = SMOTE(
            random_state=config["training"]["random_state"],
            k_neighbors=config["training"]["smote"]["k_neighbors"],
        )
        x_train, y_train = smote.fit_resample(x_train, y_train)
        logger.info("Applied SMOTE. Training size: %s", len(x_train))
    return x_train, y_train


def _train_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, dict[str, Any]],
    pd.DataFrame,
    str,
]:
    """Tune candidate models and compute holdout metrics."""
    tuned_models = tune_models(x_train, y_train, config)
    comparison: dict[str, dict[str, float]] = {}
    trained_at = datetime.now(timezone.utc).isoformat()

    for name, search in tuned_models.items():
        model = search.best_estimator_
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
        comparison[name] = compute_metrics(y_test, y_pred, y_proba)

    comparison_df = build_comparison_table(comparison)
    model_results = {
        name: build_model_result(
            name,
            comparison[name],
            search.best_params_,
            trained_at,
        )
        for name, search in tuned_models.items()
    }
    return tuned_models, comparison, model_results, comparison_df, trained_at


def _promote(
    model_results: dict[str, dict[str, Any]],
    paths: ProjectPaths,
    staging_registry_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str, ModelRegistry]:
    """Apply champion/challenger promotion logic against fresh model results."""
    registry = ModelRegistry.load(config=config, registry_path=paths.registry_db)
    registry.registry_path = staging_registry_path

    champion_name = registry.get_current_champion_name()
    if champion_name is None:
        decision = registry.evaluate_and_decide(
            model_results["logistic_regression"],
            model_results["lightgbm"],
        )
    else:
        challenger_name = registry.get_current_challenger_name()
        if challenger_name is None:
            raise ValueError("Registry challenger is not set")
        decision = registry.evaluate_and_decide(
            model_results[champion_name],
            model_results[challenger_name],
        )

    champion_name = registry.get_current_champion_name()
    if champion_name is None:
        raise ValueError("Champion model was not selected")

    logger.info(
        "Champion/challenger decision: %s | champion=%s | challenger=%s",
        decision["action"],
        registry.get_current_champion_name(),
        registry.get_current_challenger_name(),
    )
    return decision, champion_name, registry


def _evaluate(
    champion_model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: pd.DataFrame,
    staging: dict[str, Path],
    champion_name: str,
) -> dict[str, Any]:
    """Persist champion evaluation artifacts for the holdout split."""
    y_proba = champion_model.predict_proba(x_test)[:, 1]
    y_pred = champion_model.predict(x_test)
    return save_evaluation_artifacts(
        champion_model,
        x_test,
        y_test,
        y_proba,
        y_pred,
        features.columns.tolist(),
        staging["evaluation_dir"],
        champion_name,
    )


def train() -> dict:
    """Run the full training pipeline with atomic artifact publishing."""

    config = get_config()
    paths = ProjectPaths(config)
    paths.ensure_dirs()
    publisher = ArtifactPublisher(paths)
    staging_dir = publisher.create_staging_dir()
    staging = publisher.staging_paths(staging_dir)

    try:
        logger.info("Starting training pipeline")
        raw_df = load_raw_data()

        validation_report = _validate(raw_df, config)
        features, target, ids, pipeline, x_train, x_test, y_train, y_test = _engineer_features(
            raw_df,
            config,
        )
        x_train, y_train = _apply_smote(x_train, y_train, config)
        tuned_models, comparison, model_results, comparison_df, trained_at = _train_candidates(
            x_train,
            y_train,
            x_test,
            y_test,
            config,
        )
        decision, champion_name, registry = _promote(
            model_results,
            paths,
            staging["registry_db"],
            config,
        )
        champion_model = tuned_models[champion_name].best_estimator_
        eval_results = _evaluate(
            champion_model,
            x_test,
            y_test,
            features,
            staging,
            champion_name,
        )

        staging["registry_dir"].mkdir(parents=True, exist_ok=True)
        for name, search in tuned_models.items():
            joblib.dump(search.best_estimator_, staging["registry_dir"] / f"{name}.pkl")

        joblib.dump(champion_model, staging["best_model"])
        joblib.dump(pipeline.scaler, staging["scaler"])
        joblib.dump(pipeline.encoder, staging["encoder"])
        pipeline.save(str(staging["feature_pipeline"]))
        comparison_df.to_csv(staging["model_comparison"])

        metadata = {
            "trained_at": trained_at,
            "best_model_name": champion_name,
            "feature_columns": features.columns.tolist(),
            "metrics": eval_results["metrics"],
            "validation_report": validation_report,
            "best_params": tuned_models[champion_name].best_params_,
            "champion_challenger_decision": decision,
            "champion": registry.champion,
            "challenger": registry.challenger,
        }
        with staging["model_metadata"].open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        engineered_df = pipeline.feature_engineer.transform(pipeline.cleaner.transform(raw_df))
        churn_probabilities = pd.Series(champion_model.predict_proba(features)[:, 1])
        segmentation = CustomerSegmentation(config)
        segmented = segmentation.fit_predict(engineered_df, churn_probabilities)
        segmentation.save(str(staging["kmeans_model"]), str(staging["segment_mapping"]))
        segmented[["customerID", "segment", "cluster_id", "churn_probability"]].to_csv(
            staging["segment_assignments"],
            index=False,
        )

        processed_df = engineered_df.copy()
        processed_df["churn_probability"] = churn_probabilities.values
        processed_df["segment"] = segmented["segment"].values
        processed_df.to_parquet(staging["processed_data"], index=False)

        shap_background = features.sample(
            min(200, len(features)),
            random_state=config["training"]["random_state"],
        )
        from src.explainability.shap_explainer import ShapExplainer

        shap_explainer = ShapExplainer(
            champion_model,
            features.columns.tolist(),
            background_data=shap_background,
        )
        shap_explainer.save_global_summary(
            features,
            staging["evaluation_dir"] / "shap_summary.png",
        )

        publisher.publish(staging_dir)
        _log_to_mlflow(
            config,
            registry,
            decision,
            comparison,
            eval_results["metrics"],
            champion_model,
        )

        logger.info("Training pipeline completed successfully")
        return {
            "best_model_name": champion_name,
            "champion": registry.get_current_champion_name(),
            "challenger": registry.get_current_challenger_name(),
            "decision": decision,
            "metrics": eval_results["metrics"],
            "comparison": comparison,
            "staging_dir": str(staging_dir),
        }
    except Exception:
        publisher.cleanup(staging_dir)
        raise


def _log_to_mlflow(
    config: dict,
    registry: ModelRegistry,
    decision: dict,
    comparison: dict[str, dict[str, float]],
    champion_metrics: dict[str, float],
    model,
) -> None:
    """Log experiment details to MLflow."""

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    champion_name = registry.get_current_champion_name()
    with mlflow.start_run(run_name=f"champion_{champion_name}"):
        mlflow.log_param("champion_model_name", champion_name)
        mlflow.log_param("challenger_model_name", registry.get_current_challenger_name())
        mlflow.log_param("promotion_action", decision["action"])
        for metric_name, value in champion_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"champion_{metric_name}", float(value))
        for candidate, metrics in comparison.items():
            mlflow.log_metric(f"{candidate}_roc_auc", metrics["roc_auc"])
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    train()
