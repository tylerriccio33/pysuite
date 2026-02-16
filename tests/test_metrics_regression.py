"""Tests for regression metrics calculation."""

import math

import polars as pl
import pytest

from pysuite.metrics import calculate_regression_metrics


class TestBasicRegressionMetrics:
    """Test basic regression metrics are calculated correctly."""

    def test_perfect_predictions(self):
        """Test metrics when predictions are perfect."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["MAE"] == 0.0
        assert metrics["MSE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R²"] == 1.0
        assert metrics["Mean Residual"] == 0.0

    def test_constant_true_values(self):
        """Test metrics when all true values are constant."""
        y_true = pl.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred = pl.Series([5.5, 4.5, 5.0, 5.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Should still calculate MAE and other metrics
        assert metrics["MAE"] > 0.0
        assert metrics["MSE"] > 0.0
        assert metrics["RMSE"] > 0.0

    def test_single_sample(self):
        """Test metrics with single sample."""
        y_true = pl.Series([10.0])
        y_pred = pl.Series([9.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["MAE"] == 1.0
        assert metrics["MSE"] == 1.0
        assert metrics["RMSE"] == 1.0
        assert metrics["Mean Residual"] == 1.0

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # All absolute residuals are 0.5, so mean = 0.5
        assert metrics["MAE"] == 0.5

    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.5, 2.0, 2.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Sum of squared residuals is 0.5, so MSE is 0.5/3
        assert abs(metrics["MSE"] - (0.5 / 3)) < 1e-6

    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, 3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["RMSE"] == 0.0

    def test_r2_calculation(self):
        """Test R² calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["R²"] == 1.0

    def test_mean_residual(self):
        """Test Mean Residual calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([2.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # One residual of -1.0, rest are 0, so mean = -0.2
        assert metrics["Mean Residual"] == -0.2


class TestAdvancedRegressionMetrics:
    """Test advanced regression metrics."""

    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        y_true = pl.Series([100.0, 200.0, 300.0])
        y_pred = pl.Series([90.0, 200.0, 330.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Average percentage errors: 10%, 0%, 10%; MAPE ≈ 6.67%
        assert abs(metrics["MAPE"] - (0.2 / 3)) < 1e-6

    def test_mape_with_zero_values(self):
        """Test MAPE gracefully handles zero values in y_true."""
        y_true = pl.Series([0.0, 100.0, 200.0])
        y_pred = pl.Series([0.5, 90.0, 210.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Should skip the zero value
        # APE for index 1: |100-90|/100 = 10/100 = 0.1
        # APE for index 2: |200-210|/200 = 10/200 = 0.05
        # MAPE = (0.1 + 0.05) / 2 = 0.075
        assert metrics["MAPE"] == pytest.approx(0.075, abs=1e-6)

    def test_mape_all_zero_values(self):
        """Test MAPE when all y_true values are zero."""
        y_true = pl.Series([0.0, 0.0, 0.0])
        y_pred = pl.Series([1.0, 1.0, 1.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # No valid values to compute MAPE
        assert math.isinf(metrics["MAPE"])

    def test_medae_calculation(self):
        """Test Median Absolute Error calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # All absolute residuals equal 0.5, median is also 0.5
        assert metrics["MedAE"] == 0.5

    def test_max_error_calculation(self):
        """Test Maximum Absolute Error calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.0, 2.0, 5.0, 4.0, 5.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Maximum absolute error is 2.0 at index 2
        assert metrics["MaxError"] == 2.0

    def test_explained_variance(self):
        """Test Explained Variance calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Perfect fit = 1.0
        assert metrics["Explained Variance"] == 1.0

    def test_msle_with_nonnegative_values(self):
        """Test MSLE with non-negative values."""
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, 3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Perfect predictions have log difference of zero, so MSLE is zero
        assert metrics["MSLE"] == 0.0

    def test_msle_with_negative_true_values(self):
        """Test MSLE returns inf with negative y_true values."""
        y_true = pl.Series([-1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, 3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert math.isinf(metrics["MSLE"])

    def test_msle_with_negative_pred_values(self):
        """Test MSLE returns inf with negative y_pred values."""
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, -3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert math.isinf(metrics["MSLE"])

    def test_median_residual(self):
        """Test Median Residual calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Three negative, two positive residuals; median is negative
        assert metrics["Median Residual"] == -0.5

    def test_residual_std_dev(self):
        """Test Residual Standard Deviation calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0])
        y_pred = pl.Series([1.0, 2.0, 3.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # All residuals = 0, std dev = 0
        assert metrics["Residual Std Dev"] == 0.0

    def test_q1_residual(self):
        """Test Q1 (25th percentile) Residual calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # First quartile should be negative or zero
        assert metrics["Q1 Residual"] <= 0.0

    def test_q3_residual(self):
        """Test Q3 (75th percentile) Residual calculation."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Third quartile should be zero or positive
        assert metrics["Q3 Residual"] >= 0.0


class TestRegressionMetricsEdgeCases:
    """Test edge cases for regression metrics."""

    def test_all_metrics_returned(self):
        """Test that all expected metrics are returned."""
        y_true = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        expected_keys = {
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
        }

        assert set(metrics.keys()) == expected_keys

    def test_large_values(self):
        """Test metrics with large values."""
        y_true = pl.Series([1e6, 2e6, 3e6])
        y_pred = pl.Series([1.1e6, 2.1e6, 2.9e6])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Should handle large values without error
        assert not math.isnan(metrics["MAE"])
        assert not math.isnan(metrics["MSE"])
        assert not math.isnan(metrics["R²"])

    def test_very_small_values(self):
        """Test metrics with very small values."""
        y_true = pl.Series([1e-6, 2e-6, 3e-6])
        y_pred = pl.Series([1.1e-6, 2.1e-6, 2.9e-6])

        metrics = calculate_regression_metrics(y_true, y_pred)

        # Should handle small values without error
        assert not math.isnan(metrics["MAE"])
        assert not math.isnan(metrics["RMSE"])

    def test_negative_values(self):
        """Test metrics with negative values."""
        y_true = pl.Series([-5.0, -4.0, -3.0, -2.0, -1.0])
        y_pred = pl.Series([-5.5, -3.5, -3.0, -2.5, -0.5])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["MAE"] > 0.0
        assert metrics["RMSE"] > 0.0

    def test_mixed_positive_negative_values(self):
        """Test metrics with mixed positive and negative values."""
        y_true = pl.Series([-10.0, -5.0, 0.0, 5.0, 10.0])
        y_pred = pl.Series([-9.0, -6.0, 1.0, 4.0, 11.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["MAE"] > 0.0
        assert not math.isnan(metrics["R²"])
