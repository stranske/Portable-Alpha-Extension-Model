from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd

PREFERRED_INDEX_RETURN_COLUMNS = ("Monthly_TR", "Return")
SUPPORTED_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")

FREQUENCY_ALIASES = {
    "D": "daily",
    "B": "daily",
    "W": "weekly",
    "M": "monthly",
    "MS": "monthly",
    "ME": "monthly",  # Month-end frequency (pandas 2.x)
    "BM": "monthly",
    "BME": "monthly",  # Business month-end (pandas 2.x)
    "BMS": "monthly",
    "Q": "quarterly",
    "QS": "quarterly",
    "QE": "quarterly",  # Quarter-end (pandas 2.x)
    "BQ": "quarterly",
    "BQE": "quarterly",  # Business quarter-end (pandas 2.x)
    "BQS": "quarterly",
}


def infer_index_frequency(dates: pd.Series | pd.DatetimeIndex) -> str:
    """Infer date frequency from a series of timestamps or a DatetimeIndex.

    Parameters
    ----------
    dates
        Either a pd.Series containing datetime values, or a pd.DatetimeIndex.
        If a Series with numeric values and a DatetimeIndex is passed,
        the index will be used.

    Returns
    -------
    str
        One of: 'daily', 'weekly', 'monthly', 'quarterly', 'unknown', 'irregular'
    """
    # Handle DatetimeIndex directly
    if isinstance(dates, pd.DatetimeIndex):
        date_values = dates
    elif isinstance(dates, pd.Series):
        # If Series has DatetimeIndex and non-datetime values, use the index
        if isinstance(dates.index, pd.DatetimeIndex) and not pd.api.types.is_datetime64_any_dtype(
            dates
        ):
            date_values = dates.index
        else:
            # Series of datetime values
            date_values = pd.to_datetime(dates, errors="coerce")
            if date_values.isna().any():
                return "irregular"
            date_values = pd.DatetimeIndex(date_values)
    else:
        return "unknown"

    if len(date_values) < 2:
        return "unknown"
    date_values_sorted = date_values.sort_values()
    if date_values_sorted.duplicated().any():
        return "irregular"
    if len(date_values_sorted) >= 3:
        try:
            inferred = pd.infer_freq(date_values_sorted)
        except ValueError:
            inferred = None
        if inferred:
            if inferred in FREQUENCY_ALIASES:
                return FREQUENCY_ALIASES[inferred]
            if inferred.startswith("W"):
                return "weekly"
            if inferred.startswith("Q") or inferred.startswith("BQ"):
                return "quarterly"
            if inferred.startswith("M") or inferred.startswith("BM"):
                return "monthly"
    deltas = date_values_sorted.diff()[1:]  # Skip first NaT
    if len(deltas) > 0:
        delta_days = deltas.days  # TimedeltaIndex has .days directly
        if (delta_days == 1).all():
            return "daily"
        if (delta_days == 7).all():
            return "weekly"
    month_periods = date_values_sorted.to_period("M")
    month_steps = month_periods.astype(int).diff().dropna()
    if (month_steps == 1).all():
        return "monthly"
    quarter_periods = date_values_sorted.to_period("Q")
    quarter_steps = quarter_periods.astype(int).diff().dropna()
    if (quarter_steps == 1).all():
        return "quarterly"
    return "irregular"


class FrequencyValidationError(ValueError):
    """Raised when index data frequency doesn't match expected frequency."""

    def __init__(self, detected: str, expected: str, resample_hint: bool = True) -> None:
        self.detected = detected
        self.expected = expected
        hint = f" Use --resample {expected} to convert." if resample_hint else ""
        super().__init__(f"Expected {expected} data, got {detected}.{hint}")


def validate_frequency(
    series: pd.Series,
    expected: str = "monthly",
    *,
    strict: bool = True,
) -> None:
    """Validate that index series has the expected frequency.

    Parameters
    ----------
    series
        The index series. If frequency is not in attrs, will auto-detect.
    expected
        Expected frequency: "monthly", "daily", "weekly", or "quarterly".
    strict
        If True, raise FrequencyValidationError on mismatch.
        If False, emit a warning instead.

    Raises
    ------
    FrequencyValidationError
        When strict=True and frequency doesn't match expected.
    """
    detected = series.attrs.get("frequency")
    if detected is None:
        # Auto-detect if not already set
        detected = infer_index_frequency(series)
        series.attrs["frequency"] = detected

    if detected == expected:
        return
    if detected == "unknown":
        warnings.warn(
            f"Could not detect index frequency. Expected {expected}. "
            "Ensure your CSV has a Date column with consistent spacing.",
            UserWarning,
            stacklevel=2,
        )
        return
    if strict:
        raise FrequencyValidationError(detected, expected)
    warnings.warn(
        f"Index frequency mismatch: expected {expected}, detected {detected}. "
        f"Use --resample {expected} to convert or --index-frequency {detected} to skip validation.",
        UserWarning,
        stacklevel=2,
    )


def resample_to_monthly(series: pd.Series) -> pd.Series:
    """Resample a higher-frequency series to monthly.

    Uses last value of each month for returns-like data.
    Preserves series attrs including updating frequency.

    Parameters
    ----------
    series
        Input series with DatetimeIndex.

    Returns
    -------
    pd.Series
        Monthly resampled series.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Cannot resample series without DatetimeIndex")

    detected = series.attrs.get("frequency", "unknown")
    if detected == "monthly":
        return series  # Already monthly

    # For returns data, compound within each month: (1+r1)*(1+r2)*...-1
    # This preserves the total return over the period
    monthly = series.resample("ME").apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0.0)
    monthly = monthly.dropna()

    # Preserve attrs and update frequency
    monthly.attrs = series.attrs.copy()
    monthly.attrs["frequency"] = "monthly"
    monthly.attrs["resampled_from"] = detected

    return monthly


def load_parameters(path: str | Path, label_map: Dict[str, str]) -> Dict[str, Any]:
    """Return parameter dictionary by mapping CSV headers via ``label_map``.

    .. deprecated::
        CSV parameter loading is deprecated and will be removed in the next release.
        Use YAML configurations instead.
    """
    warnings.warn(
        "CSV parameter loading is deprecated and will be removed in the next release. "
        "Use YAML configurations and 'pa convert' to migrate existing CSV files.",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parameter file not found: {p}")
    try:
        df = pd.read_csv(p, comment="#")
    except (pd.errors.EmptyDataError, OSError) as exc:
        raise ValueError(f"Failed to read parameter CSV: {exc}") from exc
    data: Dict[str, Any] = {}
    if {"Parameter", "Value"}.issubset(df.columns):
        for friendly, key in label_map.items():
            match = df.loc[df["Parameter"] == friendly, "Value"]
            if not match.empty:
                val = match.iloc[0]
                if key == "risk_metrics" and isinstance(val, str):
                    data[key] = [v for v in val.split(";") if v]
                    continue
                num = cast(Any, pd.to_numeric(val, errors="coerce"))
                if pd.isna(num):
                    num = val
                if hasattr(num, "__float__") and "(%)" in friendly:
                    num = float(cast(float, num)) / 100.0
                data[key] = num
    elif not df.empty:
        row = df.iloc[0].to_dict()
        for col, key in label_map.items():
            if col in row:
                val = row[col]
                if key == "risk_metrics" and isinstance(val, str):
                    data[key] = [v for v in val.split(";") if v]
                    continue
                num = cast(Any, pd.to_numeric(val, errors="coerce"))
                if pd.isna(num):
                    num = val
                if hasattr(num, "__float__") and "(%)" in col:
                    num = float(cast(float, num)) / 100.0
                data[key] = num
    return data


def load_index_returns(path: str | Path, *, date_format: str | None = None) -> pd.Series:
    """Load index returns from a CSV file and return as Series.

    Validates and converts data to numeric format, handling non-numeric values
    by converting them to NaN and dropping them from the final series. When a
    Date column is present, the inferred frequency is stored on
    ``series.attrs["frequency"]``.
    Column selection prefers ``Monthly_TR`` then ``Return`` and raises a
    ValueError when neither column is present.
    A warning is emitted showing which column was selected and the
    available/preferred columns.
    If ``date_format`` is provided, dates are parsed strictly using that
    format.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the file cannot be read or contains no valid numeric data.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Index returns file not found: {p}")
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError) as exc:
        raise ValueError(f"Failed to read index returns CSV: {exc}") from exc

    selected_column: str | None = None
    selection_reason: str | None = None
    for col in PREFERRED_INDEX_RETURN_COLUMNS:
        if col in df.columns:
            selected_column = col
            selection_reason = "preferred column"
            raw = df[col]
            break
    if selected_column is None:
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in CSV file: {path}")
        column_list = ", ".join(map(str, df.columns))
        preferred_list = ", ".join(PREFERRED_INDEX_RETURN_COLUMNS)
        raise ValueError(
            "Expected index returns column to be one of "
            f"[{preferred_list}]. Available columns: [{column_list}]."
        )

    column_list = ", ".join(map(str, df.columns))
    preferred_list = ", ".join(PREFERRED_INDEX_RETURN_COLUMNS)
    warnings.warn(
        "Selected index returns column: "
        f"{selected_column} ({selection_reason}). "
        f"Available columns: [{column_list}]. "
        f"Preferred columns: [{preferred_list}].",
        UserWarning,
        stacklevel=2,
    )

    date_column: str | None = None
    for candidate in ("Date", "date"):
        if candidate in df.columns:
            date_column = candidate
            break

    numeric = pd.to_numeric(raw, errors="coerce")
    if date_column:
        try:
            if date_format:
                dates = pd.to_datetime(df[date_column], format=date_format, errors="raise")
            else:
                dates = pd.to_datetime(df[date_column], errors="coerce")
        except (ValueError, TypeError) as exc:
            format_hint = f" with format '{date_format}'" if date_format else ""
            raise ValueError(f"Failed to parse dates{format_hint}: {exc}") from exc
        if dates.isna().any():
            bad_values = df.loc[dates.isna(), date_column].head(3).tolist()
            format_hint = f" using format '{date_format}'" if date_format else ""
            raise ValueError(
                f"Failed to parse dates{format_hint}; invalid values: {bad_values}"
            )
        data = pd.DataFrame({"date": dates, "value": numeric})
        data = data.dropna(subset=["value", "date"])
        series = pd.Series(data["value"].to_numpy(), index=data["date"])
        series.attrs["frequency"] = infer_index_frequency(data["date"])
        series = series.sort_index()
    else:
        if date_format:
            column_list = ", ".join(map(str, df.columns))
            raise ValueError(
                f"Date format provided but no Date column found. Available columns: [{column_list}]."
            )
        series = pd.Series(numeric).dropna()
        series.attrs["frequency"] = "unknown"

    if len(series) == 0:
        raise ValueError(f"No valid numeric data found in CSV file: {path}")

    return series
