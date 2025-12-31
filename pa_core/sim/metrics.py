from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, Dict

import numpy as npt
import pandas as pd
from numpy.typing import NDArray

from ..backend import xp as np

__all__ = [
    "tracking_error",
    "value_at_risk",
    "compound",
    "annualised_return",
    "annualised_vol",
    "breach_probability",
    "breach_count",
    "conditional_value_at_risk",
    "max_drawdown",
    "time_under_water",
    "shortfall_probability",
    "summary_table",
    "register_metric",
]


_EXTRA_METRICS: Dict[str, Callable[[NDArray[npt.float64]], float]] = {}


def register_metric(name: str, func: Callable[[NDArray[npt.float64]], float]) -> None:
    """Register a custom risk metric for inclusion in ``summary_table``."""
    if name in _EXTRA_METRICS:
        raise KeyError(f"Metric already registered: {name}")
    _EXTRA_METRICS[name] = func


def _load_metric_plugins() -> None:
    for ep in entry_points(group="pa_core.risk_metrics"):
        register_metric(ep.name, ep.load())


_load_metric_plugins()


def tracking_error(strategy: NDArray[npt.float64], benchmark: NDArray[npt.float64]) -> float:
    """Return the standard deviation of active returns."""
    if strategy.shape != benchmark.shape:
        raise ValueError("shape mismatch")
    diff = np.asarray(strategy) - np.asarray(benchmark)
    if diff.size <= 1:
        return 0.0
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
    return np.cumprod(1.0 + arr, axis=1) - 1.0  # type: ignore[no-any-return]


def annualised_return(returns: NDArray[npt.float64], periods_per_year: int = 12) -> float:
    """Return annualised compound return from monthly series."""
    comp = compound(returns)
    total_return = comp[:, -1]
    years = returns.shape[1] / periods_per_year
    return float(np.power(1.0 + np.mean(total_return), 1.0 / years) - 1.0)


def annualised_vol(returns: NDArray[npt.float64], periods_per_year: int = 12) -> float:
    """Return annualised volatility from monthly returns."""
    arr = np.asarray(returns, dtype=np.float64)
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def breach_probability(
    returns: NDArray[npt.float64],
    threshold: float,
    *,
    path: int | None = None,
) -> float:
    """Return the fraction of simulation-months that breach ``threshold``.

    For 2D arrays shaped (paths, periods), this is the share of all simulated
    months across all paths that fall below ``threshold``. For 1D arrays, this
    is the share of months below the threshold. ``path`` is ignored and kept
    only for backward compatibility.
    """
    arr = np.asarray(returns, dtype=np.float64)
    return float(np.mean(arr < threshold))


def shortfall_probability(
    returns: NDArray[npt.float64],
    threshold: float = -0.05,
    *,
    compound_final: bool = True,
) -> float:
    """Return the probability of compounded returns falling below ``threshold``.

    When ``compound_final`` is ``True`` (default) the check is applied to the
    final compounded return of each simulation path.  Otherwise each monthly
    return is compared directly to the threshold.
    """

    arr = np.asarray(returns, dtype=np.float64)
    if compound_final:
        comp = compound(arr)
        final_returns = comp[:, -1] if arr.ndim > 1 else comp[[-1]]
        return float(np.mean(final_returns < threshold))
    return float(np.mean(arr < threshold))


def breach_count(returns: NDArray[npt.float64], threshold: float, *, path: int = 0) -> int:
    """Return the number of months below ``threshold`` in a selected path."""

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        series = arr
    else:
        if not (0 <= path < arr.shape[0]):
            raise IndexError("path index out of range")
        series = arr[path]
    return int(np.sum(series < threshold))


def conditional_value_at_risk(returns: NDArray[npt.float64], confidence: float = 0.95) -> float:
    """Return the conditional VaR (expected shortfall) at ``confidence``."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    flat = np.asarray(returns).reshape(-1)
    percentile = np.quantile(flat, 1 - confidence, method="lower")
    tail = flat[flat < percentile]
    if tail.size == 0:
        return float(percentile)
    return float(np.mean(tail))


def max_drawdown(returns: NDArray[npt.float64]) -> float:
    """Return the maximum drawdown from a series of arithmetic returns."""

    cumulative = np.cumsum(returns, axis=1)
    running_max = np.maximum.accumulate(cumulative, axis=1)
    drawdown = cumulative - running_max
    return float(np.min(drawdown))


def time_under_water(returns: NDArray[npt.float64]) -> float:
    """Return fraction of periods with negative compounded return."""

    comp = compound(returns)
    return float(np.mean(comp < 0.0))


def summary_table(
    returns_map: dict[str, NDArray[npt.float64]],
    *,
    periods_per_year: int = 12,
    var_conf: float = 0.95,
    breach_threshold: float = -0.02,
    shortfall_threshold: float = -0.05,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Return a summary DataFrame of key metrics for each agent.

    Parameters
    ----------
    breach_threshold:
        Monthly return threshold for :func:`breach_probability`, which reports
        the share of all simulated months across paths that breach. Defaults to
        ``-0.02`` (a 2% loss).
    shortfall_threshold:
        Threshold for :func:`shortfall_probability`.  Defaults to ``-0.05``
        (a 5% annual loss).
    """

    rows = []
    bench_arr = returns_map.get(benchmark) if benchmark else None
    for name, arr in returns_map.items():
        ann_ret = annualised_return(arr, periods_per_year)
        ann_vol = annualised_vol(arr, periods_per_year)
        var = value_at_risk(arr, confidence=var_conf)
        cvar = conditional_value_at_risk(arr, confidence=var_conf)
        breach = breach_probability(arr, breach_threshold)
        bcount = breach_count(arr, breach_threshold)
        shortfall = shortfall_probability(arr, shortfall_threshold)
        mdd = max_drawdown(arr)
        tuw = time_under_water(arr)
        te = tracking_error(arr, bench_arr) if bench_arr is not None and name != benchmark else None
        extras = {k: fn(arr) for k, fn in _EXTRA_METRICS.items()}
        rows.append(
            {
                "Agent": name,
                "AnnReturn": ann_ret,
                "AnnVol": ann_vol,
                "VaR": var,
                "CVaR": cvar,
                "MaxDD": mdd,
                "TimeUnderWater": tuw,
                "BreachProb": breach,
                "BreachCount": bcount,
                "ShortfallProb": shortfall,
                "TE": te,
                **extras,
            }
        )

    df = pd.DataFrame(rows)
    return df
