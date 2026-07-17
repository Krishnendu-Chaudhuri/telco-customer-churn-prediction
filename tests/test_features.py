from src.features.features import FeatureEngineer
from src.pipelines.cleaner import DataCleaner


def test_feature_engineering_columns(sample_df):
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    cleaned = cleaner.fit_transform(sample_df)
    engineered = engineer.transform(cleaned)

    expected = [
        "tenure_group",
        "service_count",
        "engagement_score",
        "payment_risk_score",
        "contract_risk",
        "customer_lifetime_revenue",
        "avg_monthly_value",
        "revenue_risk",
        "clv_estimate",
        "churn_propensity_score",
    ]
    for column in expected:
        assert column in engineered.columns


def test_tenure_group_labels(sample_df):
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    cleaned = cleaner.fit_transform(sample_df)
    engineered = engineer.transform(cleaned)
    assert set(engineered["tenure_group"].unique()).issubset(
        {"New", "Developing", "Stable", "Loyal"}
    )
