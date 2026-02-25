"""Flask web application for model reporting."""

import random
from typing import Any, overload

import narwhals as nw
from flask import Flask, render_template
from narwhals.typing import IntoFrame, IntoSeries

from pysuite.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_residuals,
    detect_task_type,
)


@overload
def run(
    xeval: IntoFrame, yeval: IntoSeries, ypred: IntoSeries, show: bool = ...
) -> dict[str, Any]: ...


@overload
def run(xeval: IntoFrame, yeval: str, ypred: str, show: bool = ...) -> dict[str, Any]: ...


def run(
    xeval: IntoFrame, yeval: IntoSeries | str, ypred: IntoSeries | str, show: bool = False
) -> dict[str, Any]:
    """Compute model performance metrics and table.

    Parameters
    ----------
    xeval : IntoFrame
        Feature matrix (DataFrame-like, compatible with narwhals).
        When yeval and ypred are strings, this DataFrame must contain
        those columns; they will be extracted and dropped from xeval.
    yeval : IntoSeries or str
        True target values, or column name in xeval.
    ypred : IntoSeries or str
        Predicted target values, or column name in xeval.
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
    # Handle string column references
    if isinstance(yeval, str) and isinstance(ypred, str):
        xeval_nw = nw.from_native(xeval)
        yeval_nw = xeval_nw[yeval]
        ypred_nw = xeval_nw[ypred]
        xeval_nw = xeval_nw.drop(yeval, ypred)  # type: ignore[attr-defined]
    elif isinstance(yeval, str) or isinstance(ypred, str):
        msg = "yeval and ypred must both be strings or both be Series"
        raise TypeError(msg)
    else:
        # Convert to narwhals for consistent handling
        xeval_nw = nw.from_native(xeval)
        yeval_nw = nw.from_native(yeval, series_only=True)
        ypred_nw = nw.from_native(ypred, series_only=True)

    # Check number of columns and warn if > 10
    n_cols = len(xeval_nw.columns)  # type: ignore[attr-defined]
    if n_cols > 10:
        print(
            f"Warning: {n_cols} columns detected (> 10). Consider clipping to top features for better visibility."
        )

    # Check number of rows and cap to 1000 with sampling
    n_rows = xeval_nw.shape[0]
    if n_rows > 1000:
        print(f"Auto-capping rows to 1000 (original: {n_rows}). Sampling uniformly across dataset.")
        indices = sorted(random.sample(range(n_rows), 1000))
        xeval_nw = xeval_nw[indices]
        yeval_nw = yeval_nw[indices]
        ypred_nw = ypred_nw[indices]

    # Detect task type
    task_type = detect_task_type(yeval_nw)

    # Calculate metrics
    if task_type == "regression":
        metrics = calculate_regression_metrics(yeval_nw, ypred_nw)
        # Calculate residuals only for regression
        residuals = calculate_residuals(yeval_nw, ypred_nw)
    else:
        metrics = calculate_classification_metrics(yeval_nw, ypred_nw)
        # For classification, residuals will be created from comparison
        residuals = None

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

    # Build plot data
    if task_type == "regression":
        real_list = yeval_nw.to_list()
        pred_list = ypred_nw.to_list()
        plot_data = {"real": real_list, "pred": pred_list}
    else:
        # Build confusion matrix
        real_list = yeval_nw.to_list()
        pred_list = ypred_nw.to_list()
        labels = sorted(set(real_list) | set(pred_list))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        matrix = [[0] * len(labels) for _ in labels]
        for r, p in zip(real_list, pred_list):
            matrix[label_to_idx[r]][label_to_idx[p]] += 1
        plot_data = {"labels": [str(lab) for lab in labels], "matrix": matrix}

    report_data = {
        "task_type": task_type,
        "metrics": metrics,
        "table_data": table_data,
        "columns": ordered_cols,
        "plot_data": plot_data,
    }

    if show:
        print("Launching web interface...")
        app = Flask(__name__)

        @app.route("/")
        def index() -> str:
            return render_template(
                "report.html",
                task_type=report_data["task_type"],
                metrics=report_data["metrics"],
                table_data=report_data["table_data"],
                columns=report_data["columns"],
                plot_data=report_data["plot_data"],
            )

        app.run(debug=False)

    return report_data
