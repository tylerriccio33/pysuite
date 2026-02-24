"""End-to-end tests for the model reporting web app."""

import random

import polars as pl
import pytest

from pysuite.app import run


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data.

    Returns
    -------
    tuple
        (xeval, yeval, ypred) as polars DataFrames/Series

    """
    random.seed(42)
    n_samples = 100
    n_features = 5

    # Create feature matrix
    feature_data = {
        f"feature_{i}": [random.gauss(0, 1) for _ in range(n_samples)] for i in range(n_features)
    }
    xeval = pl.DataFrame(feature_data)

    # Create predictions with some noise
    y_true = [random.gauss(0, 10) + 50 for _ in range(n_samples)]
    y_pred = [yt + random.gauss(0, 2) for yt in y_true]

    yeval = pl.Series(y_true)
    ypred = pl.Series(y_pred)

    return xeval, yeval, ypred


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data.

    Returns
    -------
    tuple
        (xeval, yeval, ypred) as polars DataFrames/Series

    """
    random.seed(42)
    n_samples = 100
    n_features = 5

    # Create feature matrix
    feature_data = {
        f"feature_{i}": [random.gauss(0, 1) for _ in range(n_samples)] for i in range(n_features)
    }
    xeval = pl.DataFrame(feature_data)

    # Create classifications using string labels to ensure classification detection
    y_true = [random.choice(["class_0", "class_1", "class_2"]) for _ in range(n_samples)]
    y_pred = [random.choice(["class_0", "class_1", "class_2"]) for _ in range(n_samples)]

    yeval = pl.Series(y_true)
    ypred = pl.Series(y_pred)

    return xeval, yeval, ypred


def test_regression_report(sample_regression_data):
    """Test that run() returns a valid regression report.

    Parameters
    ----------
    sample_regression_data : tuple
        Sample regression data fixture

    """
    xeval, yeval, ypred = sample_regression_data
    report = run(xeval, yeval, ypred)

    assert report["task_type"] == "regression"
    assert "metrics" in report
    assert "table_data" in report
    assert "columns" in report

    # Check metrics contain expected keys (both original and new metrics)
    expected_metrics = [
        "MAE",
        "MSE",
        "RMSE",
        "R²",
        "Mean Residual",
        "MAPE",
        "MedAE",
        "MaxError",
        "Explained Variance",
        "MSLE",
        "Median Residual",
        "Residual Std Dev",
        "Q1 Residual",
        "Q3 Residual",
    ]
    for metric in expected_metrics:
        assert metric in report["metrics"], f"Missing metric: {metric}"

    # Check table has data
    assert len(report["table_data"]) > 0
    assert all(col in report["table_data"][0] for col in ["pred", "real", "resid"])


def test_classification_report(sample_classification_data):
    """Test that run() returns a valid classification report.

    Parameters
    ----------
    sample_classification_data : tuple
        Sample classification data fixture

    """
    xeval, yeval, ypred = sample_classification_data
    report = run(xeval, yeval, ypred)

    assert report["task_type"] == "classification"
    assert "metrics" in report
    assert "table_data" in report
    assert "columns" in report

    # Check metrics contain expected keys (both original and new metrics)
    expected_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Micro Precision",
        "Micro Recall",
        "Micro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Balanced Accuracy",
        "Matthews Correlation Coefficient",
        "Cohen's Kappa",
    ]
    for metric in expected_metrics:
        assert metric in report["metrics"], f"Missing metric: {metric}"

    # Check table has data
    assert len(report["table_data"]) > 0
    assert all(col in report["table_data"][0] for col in ["pred", "real", "resid"])


def test_report_has_columns(sample_regression_data):
    """Test that report includes all expected columns.

    Parameters
    ----------
    sample_regression_data : tuple
        Sample regression data fixture

    """
    xeval, yeval, ypred = sample_regression_data
    report = run(xeval, yeval, ypred)

    assert "pred" in report["columns"]
    assert "real" in report["columns"]
    assert "resid" in report["columns"]
    assert any("feature_" in col for col in report["columns"])


def test_report_table_structure(sample_regression_data):
    """Test that report table has correct structure.

    Parameters
    ----------
    sample_regression_data : tuple
        Sample regression data fixture

    """
    xeval, yeval, ypred = sample_regression_data
    report = run(xeval, yeval, ypred)

    # Check each row has all columns
    for row in report["table_data"]:
        for col in report["columns"]:
            assert col in row


def test_run_with_string_columns():
    """Test run() with string column references into a single DataFrame."""
    n = 100
    df = pl.DataFrame(
        {
            "feature_0": [float(i) for i in range(n)],
            "feature_1": [float(i) * 2 for i in range(n)],
            "actual": [float(i) + 50 for i in range(n)],
            "predicted": [float(i) + 50.5 for i in range(n)],
        }
    )
    report = run(df, "actual", "predicted")

    assert report["task_type"] == "regression"
    assert "MAE" in report["metrics"]
    assert len(report["table_data"]) == n
    # actual and predicted should be extracted, not in feature columns
    assert "actual" not in report["columns"]
    assert "predicted" not in report["columns"]
    assert "feature_0" in report["columns"]
    assert "pred" in report["columns"]
    assert "real" in report["columns"]


def test_run_with_string_columns_classification():
    """Test run() with string column references for classification."""
    n = 100
    random.seed(42)
    df = pl.DataFrame(
        {
            "feature_0": [random.gauss(0, 1) for _ in range(n)],
            "y_true": [random.choice(["a", "b"]) for _ in range(n)],
            "y_pred": [random.choice(["a", "b"]) for _ in range(n)],
        }
    )
    report = run(df, "y_true", "y_pred")

    assert report["task_type"] == "classification"
    assert "Accuracy" in report["metrics"]
    assert "y_true" not in report["columns"]
    assert "y_pred" not in report["columns"]


def test_run_mixed_string_series_raises():
    """Test that mixing string and Series args raises TypeError."""
    df = pl.DataFrame({"x": [1.0], "y": [1.0]})
    with pytest.raises(TypeError, match="must both be strings"):
        run(df, "y", pl.Series([1.0]))
    with pytest.raises(TypeError, match="must both be strings"):
        run(df, pl.Series([1.0]), "y")


def test_example_regression_data():
    """Test the example regression data from main.py works correctly."""
    # Generate sample regression data (from main.py)
    n_samples = 100
    n_features = 5

    feature_data = {f"feature_{i}": [float(i) for i in range(n_samples)] for i in range(n_features)}
    xeval = pl.DataFrame(feature_data)

    y_true = [float(i) + 50 for i in range(n_samples)]
    y_pred = [yt + (i * 0.1) for i, yt in enumerate(y_true)]

    yeval = pl.Series(y_true)
    ypred = pl.Series(y_pred)

    # Test that the example data produces a valid report
    report = run(xeval, yeval, ypred)

    assert report["task_type"] == "regression"
    assert "metrics" in report
    assert "table_data" in report
    assert "columns" in report
    assert len(report["table_data"]) == n_samples
