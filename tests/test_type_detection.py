"""Tests for task type detection based on data types."""

import polars as pl

from pysuite.metrics import detect_task_type


# Numeric dtypes - should return "regression"
class TestNumericTypesRegression:
    """Test that numeric dtypes are detected as regression."""

    def test_int8_polars(self):
        """Test Int8 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int8)
        assert detect_task_type(s) == "regression"

    def test_int16_polars(self):
        """Test Int16 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int16)
        assert detect_task_type(s) == "regression"

    def test_int32_polars(self):
        """Test Int32 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32)
        assert detect_task_type(s) == "regression"

    def test_int64_polars(self):
        """Test Int64 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64)
        assert detect_task_type(s) == "regression"

    def test_uint8_polars(self):
        """Test UInt8 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt8)
        assert detect_task_type(s) == "regression"

    def test_uint16_polars(self):
        """Test UInt16 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt16)
        assert detect_task_type(s) == "regression"

    def test_uint32_polars(self):
        """Test UInt32 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt32)
        assert detect_task_type(s) == "regression"

    def test_uint64_polars(self):
        """Test UInt64 dtype returns regression with Polars."""
        s = pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt64)
        assert detect_task_type(s) == "regression"

    def test_float32_polars(self):
        """Test Float32 dtype returns regression with Polars."""
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32)
        assert detect_task_type(s) == "regression"

    def test_float64_polars(self):
        """Test Float64 dtype returns regression with Polars."""
        s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64)
        assert detect_task_type(s) == "regression"


# String dtypes - should return "classification"
class TestStringTypesClassification:
    """Test that string dtypes are detected as classification."""

    def test_string_polars(self):
        """Test String dtype returns classification with Polars."""
        s = pl.Series(["A", "B", "C", "A", "B"])
        assert detect_task_type(s) == "classification"


# Boolean dtypes - should return "classification"
class TestBooleanTypesClassification:
    """Test that boolean dtypes are detected as classification."""

    def test_boolean_polars(self):
        """Test Boolean dtype returns classification with Polars."""
        s = pl.Series([True, False, True, False], dtype=pl.Boolean)
        assert detect_task_type(s) == "classification"


# Categorical dtypes - should return "classification"
class TestCategoricalTypesClassification:
    """Test that categorical dtypes are detected as classification."""

    def test_categorical_polars(self):
        """Test Categorical dtype returns classification with Polars."""
        s = pl.Series(["A", "B", "C", "A", "B"], dtype=pl.Categorical)
        assert detect_task_type(s) == "classification"
