from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd

PREFERRED_INDEX_RETURN_COLUMNS = ("Monthly_TR", "Return")

FREQUENCY_ALIASES = {
    "D": "daily",
    "B": "daily",
    "W": "weekly",
    "M": "monthly",
    "MS": "monthly",
    "BM": "monthly",
    "BMS": "monthly",
    "Q": "quarterly",
    "QS": "quarterly",
    "BQ": "quarterly",
    "BQS": "quarterly",
}


def infer_index_frequency(dates: pd.Series) -> str:
    """Infer date frequency from a series of timestamps."""
    if len(dates) < 2:
        return "unknown"
    date_values = pd.to_datetime(dates, errors="coerce")
    if date_values.isna().any():
        return "irregular"
    date_values = date_values.sort_values()
    if date_values.duplicated().any():
        return "irregular"
    inferred = pd.infer_freq(date_values)
    if inferred:
        if inferred in FREQUENCY_ALIASES:
            return FREQUENCY_ALIASES[inferred]
        if inferred.startswith("W"):
            return "weekly"
        if inferred.startswith("Q") or inferred.startswith("BQ"):
            return "quarterly"
        if inferred.startswith("M") or inferred.startswith("BM"):
            return "monthly"
    deltas = date_values.diff().dropna()
    if (deltas.dt.days == 1).all():
        return "daily"
    if (deltas.dt.days == 7).all():
        return "weekly"
    month_periods = date_values.dt.to_period("M")
    month_steps = month_periods.astype(int).diff().dropna()
    if (month_steps == 1).all():
        return "monthly"
    quarter_periods = date_values.dt.to_period("Q")
    quarter_steps = quarter_periods.astype(int).diff().dropna()
    if (quarter_steps == 1).all():
        return "quarterly"
    return "irregular"


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
        df = pd.read_csv(p)
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


def load_index_returns(path: str | Path) -> pd.Series:
    """Load index returns from a CSV file and return as Series.

    Validates and converts data to numeric format, handling non-numeric values
    by converting them to NaN and dropping them from the final series. When a
    Date column is present, the inferred frequency is stored on
    ``series.attrs["frequency"]``.
    Column selection prefers ``Monthly_TR`` then ``Return``; otherwise the
    second column is used when present (first column for single-column files).
    A warning is emitted showing which column was selected, the reason, and
    the available/preferred columns.

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
    else:
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in CSV file: {path}")
        if df.shape[1] == 1:
            selected_column = df.columns[0]
            selection_reason = "single-column fallback"
            raw = df.iloc[:, 0]
        else:
            selected_column = df.columns[1]
            selection_reason = "second-column fallback"
            raw = df.iloc[:, 1]

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
        dates = pd.to_datetime(df[date_column], errors="coerce")
        data = pd.DataFrame({"date": dates, "value": numeric})
        data = data.dropna(subset=["value", "date"])
        series = pd.Series(data["value"].to_numpy(), index=data["date"])
        series.attrs["frequency"] = infer_index_frequency(data["date"])
    else:
        series = pd.Series(numeric).dropna()
        series.attrs["frequency"] = "unknown"

    if len(series) == 0:
        raise ValueError(f"No valid numeric data found in CSV file: {path}")

    return series
