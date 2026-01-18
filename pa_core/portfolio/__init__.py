from __future__ import annotations

from typing import TYPE_CHECKING

from .constraints import (
    COMMON_CONCENTRATION,
    COMMON_CONSTRAINTS,
    COMMON_LEVERAGE,
    COMMON_WEIGHT_BOUNDS,
    ConcentrationConstraint,
    LeverageConstraint,
    PortfolioConstraints,
    WeightBoundsConstraint,
)
from .core import DEFAULT_PORTFOLIO_EXCLUDES, compute_total_contribution_returns

if TYPE_CHECKING:
    from .aggregator import PortfolioAggregator

__all__ = [
    "COMMON_CONCENTRATION",
    "COMMON_CONSTRAINTS",
    "COMMON_LEVERAGE",
    "COMMON_WEIGHT_BOUNDS",
    "ConcentrationConstraint",
    "LeverageConstraint",
    "PortfolioAggregator",
    "PortfolioConstraints",
    "WeightBoundsConstraint",
    "compute_total_contribution_returns",
    "DEFAULT_PORTFOLIO_EXCLUDES",
]


def __getattr__(name: str) -> object:
    if name == "PortfolioAggregator":
        from .aggregator import PortfolioAggregator

        return PortfolioAggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
