"""Hyperparameter tuning utilities."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_estimators(config: dict[str, Any], random_state: int) -> dict[str, Any]:
    """Create base estimators for tuning."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight=config["models"]["logistic_regression"]["class_weight"],
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def tune_models(
    x_train,
    y_train,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune all configured models and return fitted search objects."""
    config = config or get_config()
    training_cfg = config["training"]
    random_state = training_cfg["random_state"]
    cv = StratifiedKFold(
        n_splits=training_cfg["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )

    estimators = _build_estimators(config, random_state)
    results: dict[str, Any] = {}

    for name, estimator in estimators.items():
        param_grid = config["models"][name]["param_grid"]
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=training_cfg["n_iter"],
            scoring=training_cfg["scoring"],
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )
        logger.info("Tuning model: %s", name)
        search.fit(x_train, y_train)
        results[name] = search
        logger.info(
            "Finished %s | best_score=%.4f | best_params=%s",
            name,
            search.best_score_,
            search.best_params_,
        )

    return results
