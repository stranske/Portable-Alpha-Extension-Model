"""Test the safe_to_numpy utility function."""

import numpy as np
import pandas as pd
import pytest

from pa_core.viz.utils import safe_to_numpy


def test_safe_to_numpy_with_valid_series():
    """Test safe_to_numpy with a valid numeric series."""
    series = pd.Series([1.0, 2.0, 3.0])
    result = safe_to_numpy(series)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_with_valid_dataframe():
    """Test safe_to_numpy with a valid numeric dataframe."""
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    result = safe_to_numpy(df)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_with_nan_values():
    """Test safe_to_numpy with NaN values - should use fallback."""
    series = pd.Series([1.0, np.nan, 3.0])
    result = safe_to_numpy(series, fillna_value=0.0)
    expected = np.array([1.0, 0.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_with_mixed_types():
    """Test safe_to_numpy with mixed data types - should use fallback."""
    # Create a series that would fail on direct to_numpy() due to mixed types
    # This simulates the real-world scenario that prompted the duplication issue
    series = pd.Series([1.0, 'invalid', 3.0])
    # Convert to numeric with errors='coerce' to simulate real processing
    series = pd.to_numeric(series, errors='coerce')
    
    result = safe_to_numpy(series, fillna_value=-999.0)
    expected = np.array([1.0, -999.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_with_custom_fillna_value():
    """Test safe_to_numpy with custom fillna value."""
    series = pd.Series([1.0, np.nan, 3.0])
    result = safe_to_numpy(series, fillna_value=42.0)
    expected = np.array([1.0, 42.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_empty_series():
    """Test safe_to_numpy with empty series."""
    series = pd.Series([], dtype=float)
    result = safe_to_numpy(series)
    expected = np.array([])
    np.testing.assert_array_equal(result, expected)


def test_safe_to_numpy_all_nan_series():
    """Test safe_to_numpy with series containing only NaN."""
    series = pd.Series([np.nan, np.nan])
    result = safe_to_numpy(series, fillna_value=1.0)
    expected = np.array([1.0, 1.0])
    np.testing.assert_array_equal(result, expected)