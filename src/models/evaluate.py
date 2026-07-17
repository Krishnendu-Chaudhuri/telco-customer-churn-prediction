"""Model evaluation and visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def build_comparison_table(
    model_results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Create model comparison dataframe."""
    comparison = pd.DataFrame(model_results).T
    return comparison.sort_values("roc_auc", ascending=False, na_position="last")


def build_model_result(
    model_name: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    trained_at: str,
) -> dict[str, Any]:
    """Build a champion/challenger training result payload."""
    return {
        "model_name": model_name,
        "metrics": metrics,
        "trained_at": trained_at,
        "best_params": best_params,
    }


def save_evaluation_artifacts(
    model: Any,
    x_test,
    y_test,
    y_proba,
    y_pred,
    feature_names: list[str],
    output_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Generate and save evaluation plots and reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_path = output_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
    _plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    _plot_pr_curve(y_test, y_proba, output_dir / "pr_curve.png")
    importance_df = _save_feature_importance(
        model,
        x_test,
        y_test,
        feature_names,
        output_dir,
        model_name,
    )

    metrics["classification_report_path"] = str(report_path)
    metrics["feature_importance_path"] = str(output_dir / "feature_importance.csv")
    logger.info("Saved evaluation artifacts to %s", output_dir)
    return {"metrics": metrics, "feature_importance": importance_df}


def _plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_proba, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_proba, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_feature_importance(
    model: Any,
    x_test,
    y_test,
    feature_names: list[str],
    output_dir: Path,
    model_name: str,
) -> pd.DataFrame:
    """Save model-native and permutation feature importance."""
    importance_values = None

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_).ravel()

    if importance_values is not None and len(importance_values) == len(feature_names):
        native_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_values}
        ).sort_values("importance", ascending=False)
    else:
        native_df = pd.DataFrame(columns=["feature", "importance"])

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": perm.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    native_df.to_csv(output_dir / "feature_importance.csv", index=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    if not native_df.empty:
        plt.figure(figsize=(8, 6))
        top_features = native_df.head(15)
        sns.barplot(data=top_features, x="importance", y="feature")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()

    return native_df
