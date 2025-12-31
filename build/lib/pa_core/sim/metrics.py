from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, Dict, Mapping

import pandas as pd

from ..backend import xp as np
from ..types import ArrayLike

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


_EXTRA_METRICS: Dict[str, Callable[[ArrayLike], float]] = {}


def register_metric(name: str, func: Callable[[ArrayLike], float]) -> None:
    """Register a custom risk metric for inclusion in ``summary_table``."""
    if name in _EXTRA_METRICS:
        raise KeyError(f"Metric already registered: {name}")
    _EXTRA_METRICS[name] = func


def _load_metric_plugins() -> None:
    for ep in entry_points(group="pa_core.risk_metrics"):
        register_metric(ep.name, ep.load())


_load_metric_plugins()


def tracking_error(
    strategy: ArrayLike,
    benchmark: ArrayLike,
    *,
    periods_per_year: int = 12,
) -> float:
    """Return annualised tracking error from active returns."""
    if strategy.shape != benchmark.shape:
        raise ValueError("shape mismatch")
    diff = np.asarray(strategy) - np.asarray(benchmark)
    if diff.size <= 1:
        return 0.0
    return float(np.std(diff, ddof=1) * np.sqrt(periods_per_year))


def value_at_risk(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Return the empirical VaR at the given confidence level."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    flat = np.asarray(returns).reshape(-1)
    percentile = 100 * (1 - confidence)
    return float(np.percentile(flat, percentile))


def compound(returns: ArrayLike) -> ArrayLike:
    """Return cumulative compounded returns along axis 1."""
    arr = np.asarray(returns, dtype=np.float64)
    return np.cumprod(1.0 + arr, axis=1) - 1.0  # type: ignore[no-any-return]


def annualised_return(returns: ArrayLike, periods_per_year: int = 12) -> float:
    """Return annualised compound return from monthly series."""
    comp = compound(returns)
    total_return = comp[:, -1]
    years = returns.shape[1] / periods_per_year
    return float(np.power(1.0 + np.mean(total_return), 1.0 / years) - 1.0)


def annualised_vol(returns: ArrayLike, periods_per_year: int = 12) -> float:
    """Return annualised volatility from monthly returns."""
    arr = np.asarray(returns, dtype=np.float64)
    return float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))


def breach_probability(
    returns: ArrayLike,
    threshold: float,
    *,
    path: int | None = None,
    mode: str = "month",
) -> float:
    """Return the fraction of breaches under ``threshold`` using ``mode``.

    For 2D arrays shaped (paths, periods):
    - ``mode="month"`` reports the share of all simulated months across all
      paths that fall below ``threshold`` (Option C).
    - ``mode="any"`` reports the fraction of simulation paths that breach at
      least once during the horizon (Option A).
    - ``mode="terminal"`` reports the fraction of paths that breach in the
      terminal month (Option B).
    For 1D arrays, ``mode="month"`` is the share of months below the threshold,
    while the other modes return 1.0 or 0.0 for the single path. ``path`` is
    ignored and kept only for backward compatibility.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if mode == "month":
        return float(np.mean(arr < threshold))
    if mode not in {"any", "terminal"}:
        raise ValueError('mode must be one of "month", "any", or "terminal"')
    if arr.ndim == 1:
        breached = bool(np.any(arr < threshold)) if mode == "any" else bool(arr[-1] < threshold)
        return float(breached)
    if mode == "any":
        return float(np.mean(np.any(arr < threshold, axis=1)))
    if mode == "terminal":
        return float(np.mean(arr[:, -1] < threshold))
    raise ValueError('mode must be one of "month", "any", or "terminal"')


def shortfall_probability(
    returns: ArrayLike,
    threshold: float = -0.05,
    *,
    periods_per_year: int = 12,
) -> float:
    """Return probability the terminal compounded return is below a horizon threshold.

    ``threshold`` is interpreted as an annualised return hurdle. It is converted
    to a horizon threshold based on the number of periods in ``returns``.
    """

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if arr.ndim == 1:
        arr = arr[None, :]
    years = arr.shape[1] / periods_per_year
    horizon_threshold = float(np.power(1.0 + threshold, years) - 1.0)
    comp = compound(arr)
    final_returns = comp[:, -1]
    return float(np.mean(final_returns < horizon_threshold))


def breach_count(returns: ArrayLike, threshold: float, *, path: int = 0) -> int:
    """Return the number of months below ``threshold`` in a selected path."""

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        series = arr
    else:
        if not (0 <= path < arr.shape[0]):
            raise IndexError("path index out of range")
        series = arr[path]
    return int(np.sum(series < threshold))


def conditional_value_at_risk(returns: ArrayLike, confidence: float = 0.95) -> float:
    """Return the conditional VaR (expected shortfall) at ``confidence``."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    flat = np.asarray(returns).reshape(-1)
    percentile = np.quantile(flat, 1 - confidence, method="lower")
    tail = flat[flat < percentile]
    if tail.size == 0:
        return float(percentile)
    return float(np.mean(tail))


def max_drawdown(returns: ArrayLike) -> float:
    """Return the maximum drawdown of compounded wealth paths."""

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if arr.ndim == 1:
        arr = arr[None, :]
    wealth = np.cumprod(1.0 + arr, axis=1)
    running_max = np.maximum.accumulate(wealth, axis=1)
    drawdown = wealth / running_max - 1.0
    return float(np.min(drawdown))


def time_under_water(returns: ArrayLike) -> float:
    """Return fraction of periods with negative compounded return."""

    comp = compound(returns)
    return float(np.mean(comp < 0.0))


def summary_table(
    returns_map: Mapping[str, ArrayLike],
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
        Annualised threshold for :func:`shortfall_probability`.  Defaults to
        ``-0.05`` (a 5% annual loss).
    """

    returns = returns_map
    if benchmark and "Total" not in returns_map and benchmark in returns_map:
        from ..portfolio import compute_total_contribution_returns

        total = compute_total_contribution_returns(returns_map)
        if total is not None:
            returns = dict(returns_map)
            returns["Total"] = total

    rows = []
    bench_arr = returns.get(benchmark) if benchmark else None
    bench_ann_ret = (
        annualised_return(bench_arr, periods_per_year) if bench_arr is not None else None
    )
    for name, arr in returns.items():
        ann_ret = annualised_return(arr, periods_per_year)
        ann_vol = annualised_vol(arr, periods_per_year)
        excess_return = ann_ret - bench_ann_ret if bench_ann_ret is not None else ann_ret
        var = value_at_risk(arr, confidence=var_conf)
        cvar = conditional_value_at_risk(arr, confidence=var_conf)
        breach = breach_probability(arr, breach_threshold)
        bcount = breach_count(arr, breach_threshold)
        shortfall = shortfall_probability(
            arr,
            shortfall_threshold,
            periods_per_year=periods_per_year,
        )
        mdd = max_drawdown(arr)
        tuw = time_under_water(arr)
        te = (
            tracking_error(arr, bench_arr, periods_per_year=periods_per_year)
            if bench_arr is not None and name != benchmark
            else None
        )
        extras = {k: fn(arr) for k, fn in _EXTRA_METRICS.items()}
        rows.append(
            {
                "Agent": name,
                "AnnReturn": ann_ret,
                "ExcessReturn": excess_return,
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
