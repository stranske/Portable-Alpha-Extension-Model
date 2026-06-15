from __future__ import annotations

from typing import TYPE_CHECKING

from .constraints import (
    COMMON_WEIGHT_BOUNDS,
    ConstraintValidator,
    ConstraintViolation,
    WeightBoundsConstraint,
    suggest_constraint_fixes,
)
from .core import (
    BASE_ONLY_TOTAL_WARNING,
    DEFAULT_PORTFOLIO_EXCLUDES,
    OVERLAY_SLEEVE_NAMES,
    OVERLAY_TOTAL_DESCRIPTION,
    compute_total_contribution_returns,
    is_base_only_config,
)

if TYPE_CHECKING:
    from .aggregator import PortfolioAggregator


def __getattr__(name: str) -> object:
    if name == "PortfolioAggregator":
        from .aggregator import PortfolioAggregator

        return PortfolioAggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PortfolioAggregator",
    "ConstraintValidator",
    "ConstraintViolation",
    "WeightBoundsConstraint",
    "COMMON_WEIGHT_BOUNDS",
    "suggest_constraint_fixes",
    "compute_total_contribution_returns",
    "is_base_only_config",
    "DEFAULT_PORTFOLIO_EXCLUDES",
    "OVERLAY_SLEEVE_NAMES",
    "OVERLAY_TOTAL_DESCRIPTION",
    "BASE_ONLY_TOTAL_WARNING",
]
