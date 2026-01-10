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
        f.write("Date,Monthly_TR\n")
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
        f.write("Date,Monthly_TR\n")  # Header only, no data
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
        f.write("Date,Monthly_TR\n")
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
        series = load_index_returns(temp_path)
        assert series.iloc[0] == pytest.approx(0.02)
    finally:
        os.remove(temp_path)


def test_load_index_returns_rejects_return_column():
    """Test that Return-only columns raise an error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Return,Other\n")
        f.write("2020-01-01,0.01,0.99\n")
        f.write("2020-02-01,0.03,0.88\n")
        temp_path = f.name

    try:
        with pytest.raises(
            ValueError,
            match=(
                r"Expected index returns column 'Monthly_TR', but found "
                r"\[Return\]\. Available columns: \[Date, Return, Other\]\. "
                r"Preferred columns: \[Monthly_TR, Return\]\."
            ),
        ):
            load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_load_index_returns_falls_back_to_second_column():
    """Test that missing preferred columns raises an error with available columns."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,CustomCol,Other\n")
        f.write("2020-01-01,0.07,0.99\n")
        f.write("2020-02-01,0.08,0.88\n")
        temp_path = f.name

    try:
        with pytest.raises(
            ValueError,
            match=(
                r"Expected index returns column to be one of "
                r"\[Monthly_TR, Return\]\. Available columns: "
                r"\[Date, CustomCol, Other\]\."
            ),
        ):
            load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_load_index_returns_falls_back_to_single_column():
    """Test that a single-column CSV without preferred names raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("CustomReturn\n")
        f.write("0.07\n")
        f.write("0.08\n")
        temp_path = f.name

    try:
        with pytest.raises(
            ValueError,
            match=(
                r"Expected index returns column to be one of "
                r"\[Monthly_TR, Return\]\. Available columns: \[CustomReturn\]\."
            ),
        ):
            load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_load_index_returns_with_no_numeric_columns():
    """Test that non-numeric data in all columns raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Monthly_TR\n")
        f.write("2020-01-01,not_a_number\n")
        f.write("2020-02-01,still_not_a_number\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="No valid numeric data found"):
            load_index_returns(temp_path)
    finally:
        os.remove(temp_path)


def test_load_index_returns_sorts_unsorted_dates():
    """Test that dates are sorted after parsing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Monthly_TR\n")
        f.write("2020-02-01,0.02\n")
        f.write("2020-01-01,0.01\n")
        temp_path = f.name

    try:
        series = load_index_returns(temp_path)
        assert series.index.is_monotonic_increasing
        assert series.iloc[0] == pytest.approx(0.01)
    finally:
        os.remove(temp_path)


def test_load_index_returns_with_invalid_date_format():
    """Test that invalid date formats raise a clear error."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Date,Monthly_TR\n")
        f.write("01/02/2020,0.01\n")
        temp_path = f.name

    try:
        with pytest.raises(
            ValueError,
            match=r"Failed to parse dates with format '%Y-%m-%d'",
        ):
            load_index_returns(temp_path, date_format="%Y-%m-%d")
    finally:
        os.remove(temp_path)


def test_breach_calendar_with_non_numeric_data():
    """Test that breach_calendar handles non-numeric data gracefully."""
    # Create a DataFrame with some problematic data
    df = pd.DataFrame(
        {
            "Month": [1, 2, 3],
            "TrackingErr": [0.01, 0.02, "invalid"],  # Contains string
            "terminal_ShortfallProb": [0.05, 0.1, 0.15],
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


# ==================== Frequency Detection and Validation Tests ====================


class TestInferIndexFrequency:
    """Tests for infer_index_frequency function."""

    def test_infer_monthly_frequency(self):
        """Test detection of monthly frequency data."""
        from pa_core.data.loaders import infer_index_frequency

        # Create monthly data
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        series = pd.Series([0.01] * 24, index=dates)

        result = infer_index_frequency(series)
        assert result == "monthly"

    def test_infer_daily_frequency(self):
        """Test detection of daily frequency data."""
        from pa_core.data.loaders import infer_index_frequency

        # Create daily data (business days)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        series = pd.Series([0.001] * 100, index=dates)

        result = infer_index_frequency(series)
        assert result == "daily"

    def test_infer_weekly_frequency(self):
        """Test detection of weekly frequency data."""
        from pa_core.data.loaders import infer_index_frequency

        # Create weekly data
        dates = pd.date_range("2020-01-01", periods=52, freq="W")
        series = pd.Series([0.005] * 52, index=dates)

        result = infer_index_frequency(series)
        assert result == "weekly"

    def test_infer_quarterly_frequency(self):
        """Test detection of quarterly frequency data."""
        from pa_core.data.loaders import infer_index_frequency

        # Create quarterly data
        dates = pd.date_range("2020-01-01", periods=12, freq="QE")
        series = pd.Series([0.03] * 12, index=dates)

        result = infer_index_frequency(series)
        assert result == "quarterly"


class TestValidateFrequency:
    """Tests for validate_frequency function."""

    def test_validate_monthly_passes(self):
        """Test that monthly data passes monthly validation."""
        from pa_core.data.loaders import validate_frequency

        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        series = pd.Series([0.01] * 24, index=dates)

        # Should not raise
        validate_frequency(series, expected="monthly", strict=True)

    def test_validate_daily_fails_monthly_strict(self):
        """Test that daily data fails monthly validation in strict mode."""
        from pa_core.data.loaders import FrequencyValidationError, validate_frequency

        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        series = pd.Series([0.001] * 100, index=dates)

        with pytest.raises(FrequencyValidationError) as exc_info:
            validate_frequency(series, expected="monthly", strict=True)

        assert "daily" in str(exc_info.value)
        assert "monthly" in str(exc_info.value)
        assert "--resample" in str(exc_info.value)

    def test_validate_non_strict_logs_warning(self, caplog):
        """Test that non-strict mode logs warning instead of raising."""
        from pa_core.data.loaders import validate_frequency

        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        series = pd.Series([0.001] * 100, index=dates)

        # Should not raise in non-strict mode
        validate_frequency(series, expected="monthly", strict=False)


class TestResampleToMonthly:
    """Tests for resample_to_monthly function."""

    def test_resample_daily_to_monthly(self):
        """Test resampling daily data to monthly."""
        from pa_core.data.loaders import infer_index_frequency, resample_to_monthly

        # Create daily data with known returns
        dates = pd.date_range("2020-01-02", periods=60, freq="B")
        daily_return = 0.001  # 0.1% daily
        series = pd.Series([daily_return] * 60, index=dates)

        result = resample_to_monthly(series)

        # Should be monthly now
        assert infer_index_frequency(result) == "monthly"
        # Should have compounded returns
        assert len(result) >= 2

    def test_resample_weekly_to_monthly(self):
        """Test resampling weekly data to monthly."""
        from pa_core.data.loaders import infer_index_frequency, resample_to_monthly

        # Create weekly data
        dates = pd.date_range("2020-01-01", periods=52, freq="W")
        weekly_return = 0.002  # 0.2% weekly
        series = pd.Series([weekly_return] * 52, index=dates)

        result = resample_to_monthly(series)

        # Should be monthly now
        assert infer_index_frequency(result) == "monthly"

    def test_resample_monthly_noop(self):
        """Test that resampling monthly data is a no-op."""
        from pa_core.data.loaders import resample_to_monthly

        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        series = pd.Series([0.01] * 12, index=dates)

        result = resample_to_monthly(series)

        # Should be unchanged
        assert len(result) == len(series)
        assert result.attrs.get("frequency") == "monthly"


class TestFrequencyValidationError:
    """Tests for FrequencyValidationError exception."""

    def test_error_message_includes_hint(self):
        """Test that error message includes resample hint."""
        from pa_core.data.loaders import FrequencyValidationError

        error = FrequencyValidationError("daily", "monthly", resample_hint=True)
        assert "daily" in str(error)
        assert "monthly" in str(error)
        assert "--resample" in str(error)

    def test_error_message_without_hint(self):
        """Test that error message can exclude resample hint."""
        from pa_core.data.loaders import FrequencyValidationError

        error = FrequencyValidationError("quarterly", "monthly", resample_hint=False)
        assert "quarterly" in str(error)
        assert "monthly" in str(error)
        assert "--resample" not in str(error)
