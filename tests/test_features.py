"""Tests for new features: column warnings, row capping, and web interface messaging."""

import polars as pl

from pysuite import run


def test_many_columns_warning(capsys):
    """Test warning when there are more than 10 columns."""
    # Create a DataFrame with 15 columns
    data = {f"col_{i}": [float(i) for i in range(100)] for i in range(15)}
    xeval = pl.DataFrame(data)
    yeval = pl.Series([float(i) for i in range(100)])
    ypred = pl.Series([float(i) + 0.1 for i in range(100)])

    report = run(xeval, yeval, ypred)

    captured = capsys.readouterr()
    assert "Warning: 15 columns detected (> 10)" in captured.out
    assert "Consider clipping to top features" in captured.out
    assert report["task_type"] == "regression"


def test_many_rows_capping(capsys):
    """Test that rows are capped at 1000 and message is shown."""
    # Create a DataFrame with 2000 rows
    xeval = pl.DataFrame({"x": [float(i) for i in range(2000)]})
    yeval = pl.Series([float(i) for i in range(2000)])
    ypred = pl.Series([float(i) + 0.1 for i in range(2000)])

    report = run(xeval, yeval, ypred)

    captured = capsys.readouterr()
    assert "Auto-capping rows to 1000" in captured.out
    assert "(original: 2000)" in captured.out
    assert "Sampling uniformly across dataset" in captured.out
    # The report should have exactly 1000 rows
    assert len(report["table_data"]) == 1000


def test_small_dataset_no_warnings(capsys):
    """Test that no warnings are shown for small datasets."""
    xeval = pl.DataFrame({"x": [float(i) for i in range(100)]})
    yeval = pl.Series([float(i) for i in range(100)])
    ypred = pl.Series([float(i) + 0.1 for i in range(100)])

    report = run(xeval, yeval, ypred)

    captured = capsys.readouterr()
    assert "Warning" not in captured.out
    assert "Auto-capping" not in captured.out
    assert len(report["table_data"]) == 100
