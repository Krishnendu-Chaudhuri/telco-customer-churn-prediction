import pytest


@pytest.mark.slow
def test_predictor_end_to_end(sample_customer):
    pytest.importorskip("shap")
    from src.models.predictor import ChurnPredictor

    predictor = ChurnPredictor()
    if not predictor.is_ready:
        pytest.skip("Model artifacts not available")

    predictor = ChurnPredictor().load()
    result = predictor.predict_single(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert "retention_strategy" in result
    assert "clv" in result
