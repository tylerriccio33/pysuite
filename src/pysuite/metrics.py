"""Metrics calculation and task detection for model evaluation."""

from typing import Literal, TypedDict

import narwhals as nw
from narwhals.typing import IntoFrame, IntoSeries

RegressionMetrics = TypedDict(
    "RegressionMetrics",
    {
        "MAE": float,
        "MSE": float,
        "RMSE": float,
        "R²": float,
        "Mean Residual": float,
    },
)

ClassificationMetrics = TypedDict(
    "ClassificationMetrics",
    {
        "Accuracy": float,
        "Precision": float,
        "Recall": float,
        "F1 Score": float,
    },
)


def detect_task_type(
    y: IntoSeries,
) -> Literal["regression", "classification"]:
    """Detect whether the task is regression or classification.

    Uses a heuristic based on unique value ratio and count:
    - Classification if unique values ratio < 0.05 OR unique count <= 20
    - Otherwise regression

    Parameters
    ----------
    y : IntoSeries
        Target values series

    Returns
    -------
    Literal["regression", "classification"]
        The detected task type

    """
    y_nw = nw.from_native(y, series_only=True)
    n_samples = len(y_nw)
    n_unique = y_nw.n_unique()

    unique_ratio = n_unique / n_samples if n_samples > 0 else 0

    if unique_ratio < 0.05 or n_unique <= 20:
        return "classification"
    return "regression"


@nw.narwhalify
def calculate_regression_metrics(
    y_true: IntoFrame,
    y_pred: IntoFrame,
) -> RegressionMetrics:
    """Calculate regression performance metrics.

    Parameters
    ----------
    y_true : IntoFrame
        True target values
    y_pred : IntoFrame
        Predicted target values

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, MSE, RMSE, R², and Mean Residual

    """
    # Calculate residuals
    residuals = y_true - y_pred  # type: ignore[operator]

    # Use narwhals operations for calculations
    abs_residuals = residuals.abs()
    mae = float(abs_residuals.mean())

    residuals_squared = residuals * residuals
    mse = float(residuals_squared.mean())
    rmse = float(residuals_squared.mean() ** 0.5)

    mean_residual = float(residuals.mean())

    # R² calculation
    y_mean = y_true.mean()  # type: ignore[attr-defined]
    ss_res = (residuals * residuals).sum()
    ss_tot = ((y_true - y_mean) * (y_true - y_mean)).sum()
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Mean Residual": mean_residual,
    }


@nw.narwhalify
def calculate_classification_metrics(
    y_true: IntoFrame,
    y_pred: IntoFrame,
) -> ClassificationMetrics:
    """Calculate classification performance metrics.

    Parameters
    ----------
    y_true : IntoFrame
        True target values
    y_pred : IntoFrame
        Predicted target values

    Returns
    -------
    dict[str, float]
        Dictionary containing Accuracy, Precision, Recall, and F1 Score

    """
    # Calculate accuracy
    matches = (y_true == y_pred).cast(nw.Int64)  # type: ignore[attr-defined]
    accuracy = float(matches.mean())

    # For detailed metrics, we need to work with the data
    # Convert to native for per-class calculations
    y_true_native = nw.to_native(y_true)  # type: ignore[arg-type]
    y_pred_native = nw.to_native(y_pred)  # type: ignore[arg-type]

    # Get unique classes
    if hasattr(y_true_native, "to_list"):
        y_true_list = y_true_native.to_list()
        y_pred_list = y_pred_native.to_list()
    else:
        y_true_list = list(y_true_native)
        y_pred_list = list(y_pred_native)

    classes = sorted(set(y_true_list + y_pred_list))

    precisions = []
    recalls = []
    f1_scores = []

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt == cls and yp != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "Accuracy": accuracy,
        "Precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "Recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "F1 Score": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


@nw.narwhalify
def calculate_residuals(
    y_true: IntoFrame,
    y_pred: IntoFrame,
) -> IntoFrame:
    """Calculate residuals (y_true - y_pred).

    Parameters
    ----------
    y_true : IntoFrame
        True target values
    y_pred : IntoFrame
        Predicted target values

    Returns
    -------
    IntoFrame
        Series of residuals in the native dataframe format

    """
    residuals = y_true - y_pred  # type: ignore[operator]
    return nw.to_native(residuals)
