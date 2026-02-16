"""Tests for classification metrics calculation."""

import polars as pl
import pytest

from pysuite.metrics import calculate_classification_metrics


class TestBinaryClassificationMetrics:
    """Test binary classification metrics."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = pl.Series([0, 1, 0, 1, 0])
        y_pred = pl.Series([0, 1, 0, 1, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 1.0
        assert metrics["Macro Precision"] == 1.0
        assert metrics["Macro Recall"] == 1.0
        assert metrics["Macro F1"] == 1.0
        assert metrics["Micro Precision"] == 1.0
        assert metrics["Micro Recall"] == 1.0
        assert metrics["Micro F1"] == 1.0
        assert metrics["Balanced Accuracy"] == 1.0

    def test_binary_classification_simple(self):
        """Test simple binary classification."""
        y_true = pl.Series([0, 0, 1, 1, 1])
        y_pred = pl.Series([0, 0, 0, 1, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Out of 5 samples, 4 are correctly predicted
        assert metrics["Accuracy"] == 0.8

        # For class 0: TP=2, FP=1, FN=0, precision=2/3, recall=1.0
        # For class 1: TP=2, FP=0, FN=1, precision=1.0, recall=2/3
        assert metrics["Macro Precision"] == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)
        assert metrics["Macro Recall"] == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-6)

    def test_binary_string_labels(self):
        """Test binary classification with string labels."""
        y_true = pl.Series(["cat", "dog", "cat", "dog", "cat"])
        y_pred = pl.Series(["cat", "dog", "cat", "cat", "cat"])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 0.8
        assert metrics["Micro Precision"] == metrics["Accuracy"]

    def test_binary_boolean_labels(self):
        """Test binary classification with boolean labels."""
        y_true = pl.Series([True, False, True, False, True])
        y_pred = pl.Series([True, False, True, True, True])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 0.8
        assert "Macro Precision" in metrics


class TestMultiClassClassificationMetrics:
    """Test multi-class classification metrics."""

    def test_three_class_perfect(self):
        """Test three-class classification with perfect predictions."""
        y_true = pl.Series([0, 1, 2, 0, 1, 2])
        y_pred = pl.Series([0, 1, 2, 0, 1, 2])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 1.0
        assert metrics["Macro Precision"] == 1.0
        assert metrics["Macro Recall"] == 1.0
        assert metrics["Macro F1"] == 1.0

    def test_three_class_with_errors(self):
        """Test three-class classification with some errors."""
        y_true = pl.Series([0, 0, 1, 1, 2, 2])
        y_pred = pl.Series([0, 1, 1, 2, 2, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Accuracy: 3/6 = 0.5 (indices 0, 2, 4 match)
        assert metrics["Accuracy"] == pytest.approx(0.5, abs=1e-6)
        assert "Macro F1" in metrics
        assert "Weighted F1" in metrics

    def test_five_class_classification(self):
        """Test five-class classification."""
        y_true = pl.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = pl.Series([0, 1, 2, 3, 4, 0, 1, 1, 3, 4])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Out of 10 samples, 9 are correctly predicted
        assert metrics["Accuracy"] == 0.9
        assert metrics["Macro Precision"] > 0
        assert metrics["Balanced Accuracy"] > 0

    def test_ten_class_classification(self):
        """Test ten-class classification."""
        y_true = pl.Series(list(range(10)) * 2)
        y_pred = pl.Series(list(range(10)) * 2)

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 1.0
        assert metrics["Macro F1"] == 1.0


class TestAveragingMethods:
    """Test different averaging methods for metrics."""

    def test_macro_vs_micro_averaging(self):
        """Test difference between macro and micro averaging."""
        # Highly imbalanced case
        y_true = pl.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        y_pred = pl.Series([0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Macro averages treat all classes equally
        # Micro averages weight by support
        # These should be different for imbalanced data
        assert isinstance(metrics["Macro Precision"], float)
        assert isinstance(metrics["Micro Precision"], float)
        # For this imbalanced case, they may differ
        assert "Weighted Precision" in metrics

    def test_weighted_averaging(self):
        """Test weighted averaging accounts for class imbalance."""
        # Very imbalanced: 8 class 0, 2 class 1
        y_true = pl.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_pred = pl.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # All correct, so all averages should be 1.0
        assert metrics["Weighted F1"] == 1.0
        assert metrics["Macro F1"] == 1.0


class TestImbalancedDatasets:
    """Test metrics on imbalanced datasets."""

    def test_highly_imbalanced_binary(self):
        """Test binary classification with highly imbalanced classes."""
        # 95% class 0, 5% class 1
        y_true = pl.Series([0] * 19 + [1])
        y_pred = pl.Series([0] * 19 + [0])  # Predicts all as class 0

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Accuracy is high but not meaningful
        assert metrics["Accuracy"] == 0.95
        # Balanced accuracy should be lower (average of recalls)
        assert metrics["Balanced Accuracy"] < metrics["Accuracy"]

    def test_imbalanced_multiclass(self):
        """Test multi-class with imbalanced distribution."""
        y_true = pl.Series([0] * 10 + [1] * 5 + [2] * 2)
        y_pred = pl.Series([0] * 10 + [1] * 4 + [0] * 1 + [2] * 2)

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Weighted metrics should account for class frequencies
        assert metrics["Weighted F1"] > 0
        assert metrics["Macro F1"] > 0
        # These may differ due to imbalance
        assert "Weighted Precision" in metrics


class TestCorrelationMetrics:
    """Test correlation-based metrics like MCC and Cohen's Kappa."""

    def test_mcc_perfect_classification(self):
        """Test Matthews Correlation Coefficient for perfect classification."""
        y_true = pl.Series([0, 0, 1, 1, 0, 1])
        y_pred = pl.Series([0, 0, 1, 1, 0, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Perfect classification should have MCC = 1
        assert metrics["Matthews Correlation Coefficient"] == 1.0

    def test_mcc_random_classification(self):
        """Test MCC with random predictions."""
        y_true = pl.Series([0, 1, 0, 1, 0, 1])
        y_pred = pl.Series([0, 0, 1, 1, 0, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # MCC should be in range [-1, 1]
        assert -1 <= metrics["Matthews Correlation Coefficient"] <= 1

    def test_cohens_kappa_perfect_classification(self):
        """Test Cohen's Kappa for perfect classification."""
        y_true = pl.Series([0, 0, 1, 1, 0, 1])
        y_pred = pl.Series([0, 0, 1, 1, 0, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Perfect classification should have Kappa = 1
        assert metrics["Cohen's Kappa"] == 1.0

    def test_cohens_kappa_random_classification(self):
        """Test Cohen's Kappa with random predictions."""
        y_true = pl.Series([0, 1, 0, 1, 0, 1])
        y_pred = pl.Series([1, 0, 1, 0, 1, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Kappa should be in range [-1, 1]
        assert -1 <= metrics["Cohen's Kappa"] <= 1

    def test_cohens_kappa_all_same_class(self):
        """Test Cohen's Kappa when all samples are same class."""
        y_true = pl.Series([0, 0, 0, 0, 0])
        y_pred = pl.Series([0, 0, 0, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # When expected_accuracy = 1.0, kappa is defined as 0
        assert metrics["Cohen's Kappa"] == 0.0


class TestBalancedAccuracy:
    """Test Balanced Accuracy metric."""

    def test_balanced_accuracy_perfect(self):
        """Test Balanced Accuracy with perfect predictions."""
        y_true = pl.Series([0, 0, 1, 1])
        y_pred = pl.Series([0, 0, 1, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Balanced Accuracy"] == 1.0

    def test_balanced_accuracy_lower_than_accuracy(self):
        """Test Balanced Accuracy is lower than Accuracy for imbalanced data."""
        y_true = pl.Series([0, 0, 0, 0, 1])
        y_pred = pl.Series([0, 0, 0, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # Accuracy is high but balanced accuracy should be lower
        assert metrics["Accuracy"] == 0.8
        assert metrics["Balanced Accuracy"] < metrics["Accuracy"]


class TestMetricsCompleteness:
    """Test that all expected metrics are returned."""

    def test_all_metrics_present(self):
        """Test that all expected metrics are in the result."""
        y_true = pl.Series([0, 1, 0, 1, 0])
        y_pred = pl.Series([0, 1, 1, 1, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        expected_metrics = {
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
        }

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_all_metrics_are_floats(self):
        """Test that all metrics are floats."""
        y_true = pl.Series([0, 1, 0, 1, 0])
        y_pred = pl.Series([0, 1, 1, 1, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, float), (
                f"{metric_name} is not a float: {type(metric_value)}"
            )


class TestStringLabels:
    """Test classification with string labels."""

    def test_multiclass_string_labels(self):
        """Test multi-class classification with string labels."""
        y_true = pl.Series(["red", "blue", "green", "red", "blue", "green"])
        y_pred = pl.Series(["red", "blue", "green", "red", "blue", "blue"])

        metrics = calculate_classification_metrics(y_true, y_pred)

        # 5/6 correct
        assert metrics["Accuracy"] == pytest.approx(5 / 6, abs=1e-6)
        assert all(isinstance(v, float) for v in metrics.values())


class TestNumericStringLabels:
    """Test classification with numeric labels as strings."""

    def test_numeric_string_labels(self):
        """Test that string numeric labels work correctly."""
        y_true = pl.Series(["0", "1", "2", "0", "1", "2"])
        y_pred = pl.Series(["0", "1", "2", "0", "1", "2"])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics["Accuracy"] == 1.0
        assert metrics["Macro F1"] == 1.0
