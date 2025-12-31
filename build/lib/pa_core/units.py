from __future__ import annotations

from typing import Literal

import pandas as pd

from .config import CANONICAL_RETURN_UNIT, DEFAULT_MEAN_CONVERSION, MONTHS_PER_YEAR

__all__ = ["convert_annual_series_to_monthly", "normalize_index_series"]


def convert_annual_series_to_monthly(
    series: pd.Series,
    *,
    method: Literal["simple", "geometric"] = DEFAULT_MEAN_CONVERSION,
) -> pd.Series:
    """Convert an annualized return series to monthly returns."""
    if method == "geometric":
        return (1.0 + series) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    return series / MONTHS_PER_YEAR


def normalize_index_series(idx_series: pd.Series, input_unit: str) -> pd.Series:
    """Return a monthly index series based on the configured input unit."""
    unit = input_unit or CANONICAL_RETURN_UNIT
    series = pd.Series(idx_series)
    if unit == "annual":
        return convert_annual_series_to_monthly(series)
    return series
