from src.retention_engine.recommendations import RetentionEngine


def test_retention_thresholds():
    engine = RetentionEngine()
    critical = engine.recommend(0.95, 1000.0)
    high = engine.recommend(0.75, 1000.0)
    medium = engine.recommend(0.55, 1000.0)
    low = engine.recommend(0.20, 1000.0)

    assert critical["retention_strategy"] == "Dedicated retention call"
    assert high["retention_strategy"] == "Discount offer"
    assert medium["retention_strategy"] == "Loyalty bonus"
    assert low["retention_strategy"] == "No intervention"
    assert critical["risk_level"] == "Critical"
    assert low["risk_level"] == "Low"
