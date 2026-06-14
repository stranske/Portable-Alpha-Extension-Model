from __future__ import annotations

from typing import Iterable, Mapping, TypeAlias

from ..backend import xp as np
from ..types import ArrayLike

Array: TypeAlias = ArrayLike

DEFAULT_PORTFOLIO_EXCLUDES = ("Base", "Total")
OVERLAY_SLEEVE_NAMES = ("ExternalPA", "ActiveExt", "InternalPA", "InternalBeta")
_BUILT_IN_OVERLAY_SLEEVES = ", ".join(OVERLAY_SLEEVE_NAMES)
OVERLAY_TOTAL_DESCRIPTION = (
    "Total is the overlay contribution from all non-Base, non-Total sleeves "
    f"(built-in overlays: {_BUILT_IN_OVERLAY_SLEEVES}). It excludes Base, "
    "which is the benchmark comparator."
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


def _has_positive_non_benchmark_capital(agents: Iterable[object]) -> bool:
    for agent in agents:
        params = getattr(agent, "p", None)
        if isinstance(agent, Mapping):
            name = str(agent.get("name", ""))
            raw_capital = agent.get("capital", 0.0)
        else:
            name = str(getattr(agent, "name", getattr(params, "name", "")))
            raw_capital = getattr(agent, "capital", getattr(params, "capital_mm", 0.0))
        if name in DEFAULT_PORTFOLIO_EXCLUDES:
            continue
        try:
            capital = float(raw_capital or 0.0)
        except (TypeError, ValueError):
            capital = 0.0
        if capital > 0.0:
            return True
    return False


def _agent_tuple(value: object) -> tuple[object, ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def is_base_only_config(config: object) -> bool:
    """Return true when the effective run has no non-benchmark sleeve capital."""

    raw_agents = _agent_tuple(getattr(config, "agents", ())) or ()
    try:
        from ..agents.registry import build_from_config

        built_agents = _agent_tuple(build_from_config(config))  # type: ignore[arg-type]
    except (AttributeError, FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        built_agents = None
    effective_agents = raw_agents if built_agents is None else built_agents
    return not _has_positive_non_benchmark_capital(effective_agents)
