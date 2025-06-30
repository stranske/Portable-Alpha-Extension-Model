from __future__ import annotations
from ..backend import xp as np
import numpy as npt
from numpy.typing import NDArray
import pandas as pd

__all__ = [
    "tracking_error",
    "value_at_risk",
    "compound",
    "annualised_return",
    "annualised_vol",
    "summary_table",
]


def tracking_error(strategy: NDArray[npt.float64], benchmark: NDArray[npt.float64]) -> float:
    """Return the standard deviation of active returns."""
    if strategy.shape != benchmark.shape:
        raise ValueError("shape mismatch")
    diff = np.asarray(strategy) - np.asarray(benchmark)
    return float(np.std(diff, ddof=1))


def value_at_risk(returns: NDArray[npt.float64], confidence: float = 0.95) -> float:
    """Return the empirical VaR at the given confidence level."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    flat = np.asarray(returns).reshape(-1)
    percentile = 100 * (1 - confidence)
    return float(np.percentile(flat, percentile))


def compound(returns: NDArray[npt.float64]) -> NDArray[npt.float64]:
    """Return cumulative compounded returns along axis 1."""
    arr = np.asarray(returns, dtype=np.float64)
    return np.cumprod(1.0 + arr, axis=1) - 1.0


def annualised_return(
    returns: NDArray[npt.float64], periods_per_year: int = 12
) -> float:
    """Return annualised compound return from monthly series."""
    comp = compound(returns)
    total_return = comp[:, -1]
    years = returns.shape[1] / periods_per_year
    return float(np.power(1.0 + np.mean(total_return), 1.0 / years) - 1.0)


def annualised_vol(returns: NDArray[npt.float64], periods_per_year: int = 12) -> float:
    """Return annualised volatility from monthly returns."""
    arr = np.asarray(returns, dtype=np.float64)
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def summary_table(
    returns_map: dict[str, NDArray[npt.float64]],
    *,
    periods_per_year: int = 12,
    var_conf: float = 0.95,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Return a summary DataFrame of key metrics for each agent."""

    rows = []
    bench_arr = returns_map.get(benchmark) if benchmark else None
    for name, arr in returns_map.items():
        ann_ret = annualised_return(arr, periods_per_year)
        ann_vol = annualised_vol(arr, periods_per_year)
        var = value_at_risk(arr, confidence=var_conf)
        te = (
            tracking_error(arr, bench_arr)
            if bench_arr is not None and name != benchmark
            else None
        )
        rows.append({
            "Agent": name,
            "AnnReturn": ann_ret,
            "AnnVol": ann_vol,
            "VaR": var,
            "TE": te,
        })

    df = pd.DataFrame(rows)
    return df
