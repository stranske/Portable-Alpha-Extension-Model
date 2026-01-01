from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..sim.metrics import (
    breach_count,
    breach_probability,
    conditional_value_at_risk,
    compounded_return_below_zero_fraction,
    max_cumulative_sum_drawdown,
)

Array: TypeAlias = NDArray[np.float64]


@dataclass
class RiskMetrics:
    """Container for common risk statistics."""

    cvar: float
    max_cumulative_sum_drawdown: float
    compounded_return_below_zero_fraction: float
    breach_probability: float
    breach_count: int


class RiskMetricsAgent:
    """Compute CVaR, drawdown and breach metrics from return paths."""

    def __init__(self, *, var_conf: float = 0.95, breach_threshold: float = -0.02) -> None:
        self.var_conf = var_conf
        self.breach_threshold = breach_threshold

    def run(self, returns: Array) -> RiskMetrics:
        """Return risk metrics for the given return paths."""

        cvar = conditional_value_at_risk(returns, confidence=self.var_conf)
        mdd = max_cumulative_sum_drawdown(returns)
        tuw = compounded_return_below_zero_fraction(returns)
        bprob = breach_probability(returns, self.breach_threshold)
        bcount = breach_count(returns, self.breach_threshold)
        return RiskMetrics(cvar, mdd, tuw, bprob, bcount)
