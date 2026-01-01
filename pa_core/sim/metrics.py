from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Callable, Dict, Mapping

import pandas as pd

from ..backend import xp as np
from ..types import ArrayLike
from ..units import DEFAULT_BREACH_THRESHOLD, DEFAULT_SHORTFALL_THRESHOLD

__all__ = [
    "active_return_volatility",
    "tracking_error",
    "value_at_risk",
    "compound",
    "annualised_return",
    "annualised_vol",
    "breach_probability",
    "breach_count",
    "conditional_value_at_risk",
    "max_cumulative_sum_drawdown",
    "max_drawdown",
    "compounded_return_below_zero_fraction",
    "time_under_water",
    "terminal_return_below_threshold_prob",
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


def active_return_volatility(
    strategy: ArrayLike,
    benchmark: ArrayLike,
    *,
    periods_per_year: int = 12,
) -> float:
    """Return annualised volatility of active returns (tracking error)."""
    if strategy.shape != benchmark.shape:
        raise ValueError("shape mismatch")
    diff = np.asarray(strategy) - np.asarray(benchmark)
    if diff.size <= 1:
        return 0.0
    return float(np.std(diff, ddof=1) * np.sqrt(periods_per_year))


def tracking_error(
    strategy: ArrayLike,
    benchmark: ArrayLike,
    *,
    periods_per_year: int = 12,
) -> float:
    """Deprecated alias for :func:`active_return_volatility`."""
    warnings.warn(
        "tracking_error is deprecated; use active_return_volatility",
        DeprecationWarning,
        stacklevel=2,
    )
    return active_return_volatility(
        strategy,
        benchmark,
        periods_per_year=periods_per_year,
    )


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


def terminal_return_below_threshold_prob(
    returns: ArrayLike,
    threshold: float = -0.05,
    *,
    periods_per_year: int = 12,
) -> float:
    """Return probability terminal compounded return is below a horizon threshold.

    ``threshold`` is interpreted as an annualised return hurdle. For 2D inputs,
    the probability is computed from terminal compounded returns over the full
    horizon. For 1D inputs, rolling windows of length ``periods_per_year`` are
    used to estimate the shortfall frequency.
    """

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if arr.ndim == 1:
        window = min(arr.shape[0], periods_per_year)
        years = window / periods_per_year
        horizon_threshold = float(np.power(1.0 + threshold, years) - 1.0)
        if window == 1:
            window_returns = arr
        else:
            window_returns = np.empty(arr.shape[0] - window + 1, dtype=np.float64)
            for idx in range(window_returns.size):
                window_returns[idx] = np.prod(1.0 + arr[idx : idx + window]) - 1.0
        return float(np.mean(window_returns < horizon_threshold))
    years = arr.shape[1] / periods_per_year
    horizon_threshold = float(np.power(1.0 + threshold, years) - 1.0)
    comp = compound(arr)
    final_returns = comp[:, -1]
    return float(np.mean(final_returns < horizon_threshold))


def shortfall_probability(
    returns: ArrayLike,
    threshold: float = -0.05,
    *,
    periods_per_year: int = 12,
) -> float:
    """Deprecated alias for :func:`terminal_return_below_threshold_prob`."""
    warnings.warn(
        "shortfall_probability is deprecated; use terminal_return_below_threshold_prob",
        DeprecationWarning,
        stacklevel=2,
    )
    return terminal_return_below_threshold_prob(
        returns,
        threshold,
        periods_per_year=periods_per_year,
    )


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


def max_cumulative_sum_drawdown(returns: ArrayLike) -> float:
    """Return the maximum drawdown computed from compounded wealth paths."""

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if arr.ndim == 1:
        arr = arr[None, :]
    compounded = compound(arr)
    wealth = 1.0 + compounded
    # Include the initial wealth so the first-period drop is measured correctly.
    wealth = np.concatenate([np.ones((wealth.shape[0], 1)), wealth], axis=1)
    running_max = np.maximum.accumulate(wealth, axis=1)
    drawdown = wealth / running_max - 1.0
    return float(np.min(drawdown))


def max_drawdown(returns: ArrayLike) -> float:
    """Deprecated alias for :func:`max_cumulative_sum_drawdown`."""
    warnings.warn(
        "max_drawdown is deprecated; use max_cumulative_sum_drawdown",
        DeprecationWarning,
        stacklevel=2,
    )
    return max_cumulative_sum_drawdown(returns)


def compounded_return_below_zero_fraction(returns: ArrayLike) -> float:
    """Return fraction of periods with negative compounded return."""

    comp = compound(returns)
    return float(np.mean(comp < 0.0))


def time_under_water(returns: ArrayLike) -> float:
    """Deprecated alias for :func:`compounded_return_below_zero_fraction`."""
    warnings.warn(
        "time_under_water is deprecated; use compounded_return_below_zero_fraction",
        DeprecationWarning,
        stacklevel=2,
    )
    return compounded_return_below_zero_fraction(returns)


def summary_table(
    returns_map: Mapping[str, ArrayLike],
    *,
    periods_per_year: int = 12,
    var_conf: float = 0.95,
    breach_threshold: float = DEFAULT_BREACH_THRESHOLD,
    shortfall_threshold: float = DEFAULT_SHORTFALL_THRESHOLD,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Return a summary DataFrame of key metrics for each agent.

    Parameters
    ----------
    returns_map:
        Mapping of agent name to monthly return series (shape: paths x months).
        AnnReturn, AnnVol, and TE outputs are annualised using ``periods_per_year``.
    breach_threshold:
        Monthly return threshold for :func:`breach_probability`, which reports
        the share of all simulated months across paths that breach. Defaults to
        the module-level default in :mod:`pa_core.units`.
    shortfall_threshold:
        Annualised threshold for :func:`terminal_return_below_threshold_prob`.
        Defaults to the module-level default in :mod:`pa_core.units`.
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
        shortfall = terminal_return_below_threshold_prob(
            arr,
            shortfall_threshold,
            periods_per_year=periods_per_year,
        )
        mdd = max_cumulative_sum_drawdown(arr)
        tuw = compounded_return_below_zero_fraction(arr)
        te = (
            active_return_volatility(arr, bench_arr, periods_per_year=periods_per_year)
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
