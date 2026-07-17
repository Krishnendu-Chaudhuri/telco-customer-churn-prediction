"""Explainability utilities."""

__all__ = ["ShapExplainer"]


def __getattr__(name: str):
    if name == "ShapExplainer":
        from src.explainability.shap_explainer import ShapExplainer

        return ShapExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
