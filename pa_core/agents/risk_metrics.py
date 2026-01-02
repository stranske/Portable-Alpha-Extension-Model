from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..sim.metrics import (
    breach_count_path0,
    breach_probability,
    compounded_return_below_zero_fraction,
    cvar_monthly,
    max_cumulative_sum_drawdown,
)
from ..units import DEFAULT_BREACH_THRESHOLD

Array: TypeAlias = NDArray[np.float64]


@dataclass
class RiskMetrics:
    """Container for monthly risk statistics derived from return paths.

    All fields are monthly-draw metrics; ``monthly_breach_count_path0`` is a
    path-0-only diagnostic count.
    """

    monthly_cvar: float
    monthly_max_cumulative_sum_drawdown: float
    monthly_compounded_return_below_zero_fraction: float
    monthly_breach_probability: float
    monthly_breach_count_path0: int


class RiskMetricsAgent:
    """Compute monthly CVaR, drawdown, and breach metrics from return paths."""

    def __init__(
        self, *, var_conf: float = 0.95, breach_threshold: float = DEFAULT_BREACH_THRESHOLD
    ) -> None:
        self.var_conf = var_conf
        self.breach_threshold = breach_threshold

    def run(self, returns: Array) -> RiskMetrics:
        """Return monthly risk metrics for the given return paths."""

        cvar = cvar_monthly(returns, confidence=self.var_conf)
        mdd = max_cumulative_sum_drawdown(returns)
        tuw = compounded_return_below_zero_fraction(returns)
        bprob = breach_probability(returns, self.breach_threshold)
        bcount = breach_count_path0(returns, self.breach_threshold)
        return RiskMetrics(
            monthly_cvar=cvar,
            monthly_max_cumulative_sum_drawdown=mdd,
            monthly_compounded_return_below_zero_fraction=tuw,
            monthly_breach_probability=bprob,
            monthly_breach_count_path0=bcount,
        )
