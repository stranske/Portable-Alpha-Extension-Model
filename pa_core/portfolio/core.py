from __future__ import annotations

from typing import Iterable, Mapping, TypeAlias

from ..backend import xp as np
from ..types import ArrayLike

Array: TypeAlias = ArrayLike

DEFAULT_PORTFOLIO_EXCLUDES = ("Base", "Total")
OVERLAY_SLEEVE_NAMES = ("ExternalPA", "ActiveExt", "InternalPA", "InternalBeta")
OVERLAY_TOTAL_DESCRIPTION = (
    "Total is the overlay contribution from ExternalPA, ActiveExt, InternalPA, "
    "and InternalBeta. It excludes Base, which is the benchmark comparator."
)
BASE_ONLY_TOTAL_WARNING = (
    "Base-only configuration: Total excludes Base, so Total will report zero "
    "overlay contribution rather than the index return. Use the Base row for "
    "benchmark return."
)


def compute_total_contribution_returns(
    returns_map: Mapping[str, Array],
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


def is_base_only_config(config: object) -> bool:
    """Return true when the run config has no non-benchmark sleeve capital."""

    agents = getattr(config, "agents", ()) or ()
    for agent in agents:
        name = str(getattr(agent, "name", ""))
        if name in DEFAULT_PORTFOLIO_EXCLUDES:
            continue
        try:
            capital = float(getattr(agent, "capital", 0.0) or 0.0)
        except (TypeError, ValueError):
            capital = 0.0
        if capital > 0.0:
            return False
    return True
