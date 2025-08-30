"""Test data loading validation and error handling."""

import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pa_core.data import load_index_returns
from pa_core.viz import breach_calendar, rolling_panel


def test_load_index_returns_with_malformed_data():
    """Test that load_index_returns handles malformed CSV data gracefully."""
    # Create a malformed CSV with non-numeric data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Date,Return\n")
        f.write("2020-01-01,0.01\n")
        f.write("2020-02-01,invalid_number\n")  # This should cause issues
        f.write("2020-03-01,0.03\n")
        csv_path = f.name

    try:
        # This should not crash, but handle the error gracefully
        series = load_index_returns(csv_path)
        # Check that we got a valid numeric series
        assert isinstance(series, pd.Series)
        assert len(series) > 0
        # All values should be numeric (after error handling)
        assert pd.api.types.is_numeric_dtype(series)
    finally:
        Path(csv_path).unlink()


def test_load_index_returns_with_empty_data():
    """Test that load_index_returns handles empty CSV files gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Date,Return\n")  # Header only, no data
        csv_path = f.name

    try:
        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="No valid numeric data found"):
            load_index_returns(csv_path)
    finally:
        Path(csv_path).unlink()


def test_load_index_returns_with_text_in_numeric_columns():
    """Test handling of mixed data types in numeric columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Date,Return\n")
        f.write("2020-01-01,0.01\n")
        f.write("2020-02-01,N/A\n")  # Common CSV format for missing data
        f.write("2020-03-01,0.03\n")
        f.write("2020-04-01,#DIV/0!\n")  # Excel error values
        csv_path = f.name

    try:
        series = load_index_returns(csv_path)
        # Should handle these gracefully
        assert isinstance(series, pd.Series)
        # Non-numeric values should be handled appropriately
        assert pd.api.types.is_numeric_dtype(series)
    finally:
        Path(csv_path).unlink()


def test_breach_calendar_with_non_numeric_data():
    """Test that breach_calendar handles non-numeric data gracefully."""
    # Create a DataFrame with some problematic data
    df = pd.DataFrame({
        'Month': [1, 2, 3],
        'TrackingErr': [0.01, 0.02, "invalid"],  # Contains string
        'ShortfallProb': [0.05, 0.1, 0.15]
    })

    # This should not crash after our fix
    fig = breach_calendar.make(df)
    assert fig is not None


def test_rolling_panel_with_malformed_data():
    """Test that rolling_panel handles malformed input data gracefully."""
    # Create array with some NaN values
    data = np.array([
        [0.01, np.nan, 0.03, 0.04],
        [0.02, 0.025, np.inf, 0.035],  # Contains infinity
        [-0.01, 0.015, 0.025, 0.045]
    ])

    try:
        fig = rolling_panel.make(data, window=2)
        assert fig is not None
    except Exception as e:
        # Should handle numeric issues gracefully
        assert "cannot convert" in str(e).lower() or "invalid" in str(e).lower()