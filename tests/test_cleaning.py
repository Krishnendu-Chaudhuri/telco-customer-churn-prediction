import pandas as pd

from src.pipelines.cleaner import DataCleaner


def test_total_charges_blank_handling():
    cleaner = DataCleaner()
    df = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "SeniorCitizen": [0, 0],
            "tenure": [1, 10],
            "MonthlyCharges": [20.0, 50.0],
            "TotalCharges": [" ", "500.0"],
        }
    )
    cleaned = cleaner.fit_transform(df)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["TotalCharges"].dtype.kind in {"f", "i"}
