from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_parameters(path: str | Path, label_map: Dict[str, str]) -> Dict[str, Any]:
    """Return parameter dictionary by mapping CSV headers via ``label_map``."""
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
                try:
                    num = pd.to_numeric(val)
                except Exception:
                    num = val
                if hasattr(num, "__float__") and "(%)" in friendly:
                    num = float(num) / 100.0
                data[key] = num
    elif not df.empty:
        row = df.iloc[0].to_dict()
        for col, key in label_map.items():
            if col in row:
                val = row[col]
                if key == "risk_metrics" and isinstance(val, str):
                    data[key] = [v for v in val.split(";") if v]
                    continue
                try:
                    num = pd.to_numeric(val)
                except Exception:
                    num = val
                if hasattr(num, "__float__") and "(%)" in col:
                    num = float(num) / 100.0
                data[key] = num
    return data


def load_index_returns(path: str | Path) -> pd.Series:
    """Load index returns from a CSV file and return as Series."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df.iloc[:, 1]
