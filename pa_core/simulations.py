"""Legacy wrapper module for simulation utilities."""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Iterable, Tuple

from .agents import (
    ActiveExtensionAgent,
    Agent,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from .backend import xp as np
from .portfolio import compute_total_contribution_returns
from .sim import (
    draw_financing_series,
    draw_joint_returns,
    prepare_mc_universe,
    simulate_alpha_streams,
    simulate_financing,
)
from .types import ArrayLike

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
    "compute_total_returns",
    "simulate_agents",
]


@singledispatch
def _resolve_streams(
    agent: Agent,
    r_beta: ArrayLike,
    r_H: ArrayLike,
    r_E: ArrayLike,
    r_M: ArrayLike,
    f_int: ArrayLike,
    f_ext_pa: ArrayLike,
    f_act_ext: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Return ``(alpha_stream, financing)`` for ``agent``."""
    raise TypeError(f"Unsupported agent type: {type(agent)}")


@_resolve_streams.register
def _(agent: BaseAgent, *streams: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_H, f_int


@_resolve_streams.register
def _(agent: ExternalPAAgent, *streams: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_M, f_ext_pa


@_resolve_streams.register
def _(agent: ActiveExtensionAgent, *streams: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_E, f_act_ext


@_resolve_streams.register
def _(agent: InternalPAAgent, *streams: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_H, np.zeros_like(r_beta)


@_resolve_streams.register
def _(agent: InternalBetaAgent, *streams: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return np.zeros_like(r_beta), f_int


def simulate_agents(
    agents: Iterable[Agent],
    r_beta: ArrayLike,
    r_H: ArrayLike,
    r_E: ArrayLike,
    r_M: ArrayLike,
    f_int: ArrayLike,
    f_ext_pa: ArrayLike,
    f_act_ext: ArrayLike,
) -> dict[str, ArrayLike]:
    """Return per-agent monthly returns using vectorised operations."""
    results: dict[str, ArrayLike] = {}
    streams = (r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext)
    for agent in agents:
        alpha, financing = _resolve_streams(agent, *streams)
        results[agent.p.name] = agent.monthly_returns(r_beta, alpha, financing)

    total = compute_total_returns(results)
    if total is not None:
        results["Total"] = total
    return results


def compute_total_returns(
    returns_map: dict[str, ArrayLike],
    *,
    exclude: Iterable[str] = ("Base", "Total"),
) -> ArrayLike | None:
    """Return Total portfolio returns as the sum of contribution sleeves."""
    return compute_total_contribution_returns(returns_map, exclude=exclude)
