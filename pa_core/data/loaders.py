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
        stacklevel=2
    )
    df = pd.read_csv(path)
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
    """Load index returns from a CSV file and return as Series."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df.iloc[:, 1]
