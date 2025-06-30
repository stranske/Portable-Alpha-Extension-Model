from __future__ import annotations
from .backend import xp as np
import numpy as npt
from numpy.typing import NDArray

__all__ = ["tracking_error", "value_at_risk"]


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
