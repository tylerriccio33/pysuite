"""Flask web application for model reporting."""

from typing import Any

import narwhals as nw
from flask import Flask, render_template
from narwhals.typing import IntoFrame, IntoSeries

from pysuite.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_residuals,
    detect_task_type,
)


def run(
    xeval: IntoFrame, yeval: IntoSeries, ypred: IntoSeries, show: bool = False
) -> dict[str, Any]:
    """Compute model performance metrics and table.

    Parameters
    ----------
    xeval : IntoFrame
        Feature matrix (DataFrame-like, compatible with narwhals)
    yeval : IntoSeries
        True target values (Series-like, compatible with narwhals)
    ypred : IntoSeries
        Predicted target values (Series-like, compatible with narwhals)
    show : bool, optional
        If True, launch Flask web interface. Default is False.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - task_type: "regression" or "classification"
        - metrics: Dict of performance metrics
        - table_data: List of dicts with predictions, actuals, residuals, and features
        - columns: Ordered list of column names

    Examples
    --------
    >>> import polars as pl
    >>> xeval = pl.DataFrame({"x": [float(i) for i in range(100)]})
    >>> yeval = pl.Series([float(i) for i in range(100)])
    >>> ypred = pl.Series([float(i) + 0.5 for i in range(100)])
    >>> report = run(xeval, yeval, ypred)
    >>> report["task_type"]
    'regression'
    >>> "MAE" in report["metrics"]
    True
    >>> len(report["table_data"]) == 100
    True
    >>> set(report["columns"]) == {"pred", "real", "resid", "x"}
    True

    """
    # Detect task type
    task_type = detect_task_type(yeval)

    # Calculate metrics
    if task_type == "regression":
        metrics = calculate_regression_metrics(yeval, ypred)
        # Calculate residuals only for regression
        residuals = calculate_residuals(yeval, ypred)
    else:
        metrics = calculate_classification_metrics(yeval, ypred)
        # For classification, residuals will be created from comparison
        residuals = None

    # Convert to narwhals for consistent handling
    xeval_nw = nw.from_native(xeval)
    yeval_nw = nw.from_native(yeval, series_only=True)
    ypred_nw = nw.from_native(ypred, series_only=True)

    # Create residuals for classification if not already created
    if residuals is None:
        residuals_nw = (yeval_nw != ypred_nw).cast(nw.Int64)
    else:
        residuals_nw = nw.from_native(residuals, series_only=True)

    # Build result table using narwhals
    result_table = xeval_nw.with_columns(  # type: ignore[attr-defined]
        [
            ypred_nw.alias("pred"),
            yeval_nw.alias("real"),
            residuals_nw.alias("resid"),
        ]
    )

    # Reorder columns: pred, real, resid, then features
    feature_cols = list(xeval_nw.columns)  # type: ignore[attr-defined]
    ordered_cols = ["pred", "real", "resid"] + feature_cols
    result_table = result_table.select(ordered_cols)

    # Convert to list of dicts for template
    native_result = result_table.to_native()
    if hasattr(native_result, "to_dicts"):
        table_data = native_result.to_dicts()
    else:
        table_data = native_result.to_dict(orient="records")

    report_data = {
        "task_type": task_type,
        "metrics": metrics,
        "table_data": table_data,
        "columns": ordered_cols,
    }

    if show:
        app = Flask(__name__)

        @app.route("/")
        def index() -> str:
            return render_template(
                "report.html",
                task_type=report_data["task_type"],
                metrics=report_data["metrics"],
                table_data=report_data["table_data"],
                columns=report_data["columns"],
            )

        app.run(debug=False)

    return report_data
