"""Test data loading validation and error handling."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pa_core.data import load_index_returns
from pa_core.viz import breach_calendar, rolling_panel


def test_load_index_returns_with_malformed_data():
    """Test that load_index_returns handles malformed CSV data gracefully."""
    # Create a malformed CSV with non-numeric data
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return\n")
        f.write("2020-01-01,0.01\n")
        f.write("2020-02-01,invalid_number\n")  # This should cause issues
        f.write("2020-03-01,0.03\n")
        temp_path = f.name

    try:
        # This should not crash, but handle the error gracefully
        series = load_index_returns(temp_path)
        # Check that we got a valid numeric series
        assert isinstance(series, pd.Series)
        assert len(series) > 0
        # All values should be numeric (after error handling)
        assert pd.api.types.is_numeric_dtype(series)
    finally:
        os.remove(temp_path)


def test_load_index_returns_with_empty_data():
    """Test that load_index_returns handles empty CSV files gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return\n")  # Header only, no data
        temp_path = f.name

    try:
        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="No valid numeric data found"):
            load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_load_index_returns_with_text_in_numeric_columns():
    """Test handling of mixed data types in numeric columns."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return\n")
        f.write("2020-01-01,0.01\n")
        f.write("2020-02-01,N/A\n")  # Common CSV format for missing data
        f.write("2020-03-01,0.03\n")
        f.write("2020-04-01,#DIV/0!\n")  # Excel error values
        temp_path = f.name

    try:
        series = load_index_returns(temp_path)
        # Should handle these gracefully
        assert isinstance(series, pd.Series)
        # Non-numeric values should be handled appropriately
        assert pd.api.types.is_numeric_dtype(series)
    finally:
        os.remove(temp_path)


def test_load_index_returns_prefers_monthly_tr_column():
    """Test that Monthly_TR is preferred when present."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return,Monthly_TR\n")
        f.write("2020-01-01,0.01,0.02\n")
        f.write("2020-02-01,0.03,0.04\n")
        temp_path = f.name

    try:
        with pytest.warns(
            UserWarning,
            match=(
                r"Selected index returns column: Monthly_TR "
                r"\(preferred column\); available columns:"
            ),
        ):
            series = load_index_returns(temp_path)
        assert series.iloc[0] == pytest.approx(0.02)
    finally:
        os.remove(temp_path)


def test_load_index_returns_prefers_return_column():
    """Test that Return is selected when Monthly_TR is absent."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return,Other\n")
        f.write("2020-01-01,0.01,0.99\n")
        f.write("2020-02-01,0.03,0.88\n")
        temp_path = f.name

    try:
        with pytest.warns(
            UserWarning,
            match=(
                r"Selected index returns column: Return "
                r"\(preferred column\); available columns:"
            ),
        ):
            series = load_index_returns(temp_path)
        assert series.iloc[0] == pytest.approx(0.01)
    finally:
        os.remove(temp_path)


def test_load_index_returns_falls_back_to_second_column():
    """Test that the second column is used when no standard names exist."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,CustomCol,Other\n")
        f.write("2020-01-01,0.07,0.99\n")
        f.write("2020-02-01,0.08,0.88\n")
        temp_path = f.name

    try:
        with pytest.warns(
            UserWarning,
            match=(
                r"Selected index returns column: CustomCol "
                r"\(second-column fallback\); available columns:"
            ),
        ):
            series = load_index_returns(temp_path)
        assert series.iloc[0] == pytest.approx(0.07)
    finally:
        os.remove(temp_path)


def test_load_index_returns_falls_back_to_single_column():
    """Test that the only column is used when the CSV has one column."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("CustomReturn\n")
        f.write("0.07\n")
        f.write("0.08\n")
        temp_path = f.name

    try:
        with pytest.warns(
            UserWarning,
            match=(
                r"Selected index returns column: CustomReturn "
                r"\(single-column fallback\); available columns:"
            ),
        ):
            series = load_index_returns(temp_path)
        assert series.iloc[0] == pytest.approx(0.07)
    finally:
        os.remove(temp_path)


def test_load_index_returns_with_no_numeric_columns():
    """Test that non-numeric data in all columns raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Label\n")
        f.write("2020-01-01,not_a_number\n")
        f.write("2020-02-01,still_not_a_number\n")
        temp_path = f.name

    try:
        with pytest.warns(
            UserWarning,
            match=(
                r"Selected index returns column: Label "
                r"\(second-column fallback\); available columns:"
            ),
        ):
            with pytest.raises(ValueError, match="No valid numeric data found"):
                load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_breach_calendar_with_non_numeric_data():
    """Test that breach_calendar handles non-numeric data gracefully."""
    # Create a DataFrame with some problematic data
    df = pd.DataFrame(
        {
            "Month": [1, 2, 3],
            "TrackingErr": [0.01, 0.02, "invalid"],  # Contains string
            "ShortfallProb": [0.05, 0.1, 0.15],
        }
    )

    # This should not crash after our fix
    fig = breach_calendar.make(df)
    assert fig is not None


def test_rolling_panel_with_malformed_data():
    """Test that rolling_panel handles malformed input data gracefully."""
    # Create array with some NaN values
    data = np.array(
        [
            [0.01, np.nan, 0.03, 0.04],
            [0.02, 0.025, np.inf, 0.035],  # Contains infinity
            [-0.01, 0.015, 0.025, 0.045],
        ]
    )

    try:
        fig = rolling_panel.make(data, window=2)
        assert fig is not None
    except Exception as e:
        # Should handle numeric issues gracefully
        assert "cannot convert" in str(e).lower() or "invalid" in str(e).lower()
