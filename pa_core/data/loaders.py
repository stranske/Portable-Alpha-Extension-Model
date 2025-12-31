from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd


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
    by converting them to NaN and dropping them from the final series.
    Column selection prefers ``Monthly_TR`` then ``Return``; otherwise the
    second column is used when present (first column for single-column files).
    A warning is emitted showing which column was selected.

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
    for col in ("Monthly_TR", "Return"):
        if col in df.columns:
            selected_column = col
            raw = df[col]
            break
    else:
        if df.shape[1] == 0:
            raise ValueError(f"No columns found in CSV file: {path}")
        if df.shape[1] == 1:
            selected_column = df.columns[0]
            raw = df.iloc[:, 0]
        else:
            selected_column = df.columns[1]
            raw = df.iloc[:, 1]

    warnings.warn(
        f"Selected index returns column: {selected_column}",
        UserWarning,
        stacklevel=2,
    )

    # Convert to numeric, coerce errors to NaN, then wrap as Series to satisfy typing
    numeric = pd.to_numeric(raw, errors="coerce")
    series = pd.Series(numeric).dropna()

    if len(series) == 0:
        raise ValueError(f"No valid numeric data found in CSV file: {path}")

    return series
