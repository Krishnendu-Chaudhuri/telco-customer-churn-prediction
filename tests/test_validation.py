from src.pipelines.validator import DataValidator


def test_validation_report_structure(sample_df):
    validator = DataValidator()
    report = validator.validate(sample_df)
    assert "row_count" in report
    assert "missing_values" in report
    assert "duplicates" in report
    assert report["row_count"] == len(sample_df)
