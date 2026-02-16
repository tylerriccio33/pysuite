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
        "MAPE": float,
        "MedAE": float,
        "MaxError": float,
        "Explained Variance": float,
        "MSLE": float,
        "Median Residual": float,
        "Residual Std Dev": float,
        "Q1 Residual": float,
        "Q3 Residual": float,
    },
)

ClassificationMetrics = TypedDict(
    "ClassificationMetrics",
    {
        "Accuracy": float,
        "Precision": float,
        "Recall": float,
        "F1 Score": float,
        "Macro Precision": float,
        "Macro Recall": float,
        "Macro F1": float,
        "Micro Precision": float,
        "Micro Recall": float,
        "Micro F1": float,
        "Weighted Precision": float,
        "Weighted Recall": float,
        "Weighted F1": float,
        "Balanced Accuracy": float,
        "Matthews Correlation Coefficient": float,
        "Cohen's Kappa": float,
    },
    total=False,
)


def detect_task_type(
    y: IntoSeries,
) -> Literal["regression", "classification"]:
    """Detect whether the task is regression or classification.

    Uses dtype-based detection:
    - Numeric dtypes (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64) → regression
    - String, Boolean, Categorical, Enum → classification

    NOTE: This is a breaking change from the previous heuristic-based detection.
    Previously used unique value ratio < 0.05 or unique count <= 20 for classification.
    Now relies on data type instead of value distribution.

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
    dtype = y_nw.dtype

    # Define numeric types that indicate regression
    numeric_types = (
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.Float32,
        nw.Float64,
    )

    if dtype in numeric_types:
        return "regression"
    else:
        return "classification"


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
        Dictionary containing MAE, MSE, RMSE, R², Mean Residual, MAPE, MedAE,
        MaxError, Explained Variance, MSLE, Median Residual, Residual Std Dev,
        Q1 Residual, and Q3 Residual

    Notes
    -----
    - MAPE skips values where y_true is 0 to avoid division issues
    - MSLE requires non-negative values; returns inf if negatives are found

    """
    # Calculate residuals
    residuals = y_true - y_pred  # type: ignore[operator]

    # Basic metrics
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

    # MAPE: Mean Absolute Percentage Error
    # Skip zero values in y_true to avoid division by zero
    nonzero_mask = y_true != 0
    filtered_abs_residuals = abs_residuals.filter(nonzero_mask)
    filtered_y_true = y_true.filter(nonzero_mask)  # type: ignore[attr-defined]
    mape_values = filtered_abs_residuals / filtered_y_true.abs()
    mape = float(mape_values.mean()) if len(mape_values) > 0 else float("inf")

    # MedAE: Median Absolute Error
    med_ae = float(abs_residuals.median())

    # MaxError: Maximum Absolute Error
    max_error = float(abs_residuals.max())

    # Explained Variance
    explained_variance = (
        float(1 - (residuals_squared.mean() / ((y_true - y_mean) * (y_true - y_mean)).mean()))
        if ((y_true - y_mean) * (y_true - y_mean)).mean() != 0
        else 0.0
    )

    # MSLE: Mean Squared Log Error (requires non-negative values)
    # Check if any values are negative
    has_negative = (y_true < 0).any() or (y_pred < 0).any()  # type: ignore[attr-defined]
    if has_negative:
        msle = float("inf")
    else:
        log_true = (y_true + 1).log()  # type: ignore[attr-defined]
        log_pred = (y_pred + 1).log()  # type: ignore[attr-defined]
        msle = float(((log_true - log_pred) * (log_true - log_pred)).mean())

    # Median Residual
    median_residual = float(residuals.median())

    # Residual Std Dev
    residual_mean = residuals.mean()
    residual_variance = ((residuals - residual_mean) * (residuals - residual_mean)).mean()
    residual_std = float(residual_variance**0.5)

    # Quantiles for residuals
    q1_residual = float(residuals.quantile(0.25, interpolation="linear"))
    q3_residual = float(residuals.quantile(0.75, interpolation="linear"))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Mean Residual": mean_residual,
        "MAPE": mape,
        "MedAE": med_ae,
        "MaxError": max_error,
        "Explained Variance": explained_variance,
        "MSLE": msle,
        "Median Residual": median_residual,
        "Residual Std Dev": residual_std,
        "Q1 Residual": q1_residual,
        "Q3 Residual": q3_residual,
    }


@nw.narwhalify
def calculate_classification_metrics(
    y_true: IntoFrame,
    y_pred: IntoFrame,
) -> ClassificationMetrics:
    """Calculate comprehensive classification performance metrics.

    Parameters
    ----------
    y_true : IntoFrame
        True target values
    y_pred : IntoFrame
        Predicted target values

    Returns
    -------
    dict[str, float]
        Dictionary containing Accuracy, Precision, Recall, F1 Score, and additional metrics
        including macro/micro/weighted averages, Balanced Accuracy, MCC, and Cohen's Kappa

    """
    # Calculate accuracy
    matches = (y_true == y_pred).cast(nw.Int64)  # type: ignore[attr-defined]
    accuracy = float(matches.mean())

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
    n_samples = len(y_true_list)

    # Calculate per-class metrics
    class_metrics: dict[
        str, tuple[float, float, float, int]
    ] = {}  # {class: (precision, recall, f1, support)}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if yt == cls and yp != cls)
        support = tp + fn  # Total positive instances of this class

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[str(cls)] = (precision, recall, f1, support)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro-averaged metrics (simple average)
    precisions = [m[0] for m in class_metrics.values()]
    recalls = [m[1] for m in class_metrics.values()]
    f1_scores = [m[2] for m in class_metrics.values()]

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Micro-averaged metrics (pool all TP/FP/FN)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Weighted-averaged metrics (weight by support)
    supports = [m[3] for m in class_metrics.values()]
    total_support = sum(supports)

    weighted_precision = (
        sum(m[0] * m[3] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )
    weighted_recall = (
        sum(m[1] * m[3] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )
    weighted_f1 = (
        sum(m[2] * m[3] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )

    # Balanced Accuracy (average of per-class recalls)
    balanced_accuracy = macro_recall

    # Cohen's Kappa: (accuracy - expected_accuracy) / (1 - expected_accuracy)
    expected_accuracy = 0.0
    for cls in classes:
        p_cls_true = sum(1 for yt in y_true_list if yt == cls) / n_samples
        p_cls_pred = sum(1 for yp in y_pred_list if yp == cls) / n_samples
        expected_accuracy += p_cls_true * p_cls_pred

    cohens_kappa = (
        (accuracy - expected_accuracy) / (1 - expected_accuracy) if expected_accuracy < 1 else 0.0
    )

    # Matthews Correlation Coefficient
    # For multi-class, use the relationship with Cohen's Kappa
    # MCC can be derived from the confusion matrix, but for simplicity,
    # we use the fact that MCC relates to how much better than random the classifier is
    # For perfect classification, MCC = 1; for random, MCC ≈ 0
    if accuracy >= 1.0 and expected_accuracy < 1.0:
        mcc = 1.0
    elif accuracy <= expected_accuracy:
        mcc = 0.0
    else:
        # Use a normalized form of the correlation
        mcc = (cohens_kappa + 1) / 2 if cohens_kappa > -1 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": macro_precision,
        "Recall": macro_recall,
        "F1 Score": macro_f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F1": micro_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1,
        "Balanced Accuracy": balanced_accuracy,
        "Matthews Correlation Coefficient": mcc,
        "Cohen's Kappa": cohens_kappa,
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
