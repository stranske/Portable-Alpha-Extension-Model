from __future__ import annotations

from typing import Iterable, TypeAlias

from ..backend import xp as np
from ..types import ArrayLike
from .aggregator import PortfolioAggregator

Array: TypeAlias = ArrayLike

DEFAULT_PORTFOLIO_EXCLUDES = ("Base", "Total")


def compute_total_contribution_returns(
    returns_map: dict[str, Array],
    *,
    exclude: Iterable[str] = DEFAULT_PORTFOLIO_EXCLUDES,
) -> Array | None:
    """Return Total portfolio returns from contribution-style sleeve outputs.

    Sleeves emit contribution returns already scaled by capital share. The
    benchmark sleeve (``Base``) is excluded, and ``Total`` is computed once
    as the sum of the remaining sleeves.
    """
    total = None
    for name, arr in returns_map.items():
        if name in exclude:
            continue
        if total is None:
            total = np.zeros_like(arr)
        total = total + arr
    return total


__all__ = [
    "PortfolioAggregator",
    "compute_total_contribution_returns",
    "DEFAULT_PORTFOLIO_EXCLUDES",
]
