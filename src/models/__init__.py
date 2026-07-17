"""Model training and inference modules."""

__all__ = ["ChurnPredictor"]


def __getattr__(name: str):
    if name == "ChurnPredictor":
        from src.models.predictor import ChurnPredictor

        return ChurnPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
