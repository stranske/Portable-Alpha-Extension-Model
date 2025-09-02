"""Legacy wrapper module for simulation utilities."""

from __future__ import annotations

from typing import Any, Iterable, Tuple
from functools import singledispatch

from numpy.typing import NDArray

from .agents import (
    ActiveExtensionAgent,
    Agent,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from .backend import xp as np
from .sim import (
    draw_financing_series,
    draw_joint_returns,
    prepare_mc_universe,
    simulate_alpha_streams,
    simulate_financing,
)

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
    "simulate_agents",
]


@singledispatch
def _resolve_streams(
    agent: Agent,
    r_beta: NDArray[Any],
    r_H: NDArray[Any],
    r_E: NDArray[Any],
    r_M: NDArray[Any],
    f_int: NDArray[Any],
    f_ext_pa: NDArray[Any],
    f_act_ext: NDArray[Any],
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Return ``(alpha_stream, financing)`` for ``agent``."""
    raise TypeError(f"Unsupported agent type: {type(agent)}")


@_resolve_streams.register
def _(agent: BaseAgent, *streams: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_H, f_int


@_resolve_streams.register
def _(agent: ExternalPAAgent, *streams: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_M, f_ext_pa


@_resolve_streams.register
def _(agent: ActiveExtensionAgent, *streams: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_E, f_act_ext


@_resolve_streams.register
def _(agent: InternalBetaAgent, *streams: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_H, f_int


@_resolve_streams.register
def _(agent: InternalPAAgent, *streams: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
    r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext = streams
    return r_H, np.zeros_like(r_beta)


def simulate_agents(
    agents: Iterable[Agent],
    r_beta: NDArray[Any],
    r_H: NDArray[Any],
    r_E: NDArray[Any],
    r_M: NDArray[Any],
    f_int: NDArray[Any],
    f_ext_pa: NDArray[Any],
    f_act_ext: NDArray[Any],
) -> dict[str, NDArray[Any]]:
    """Return per-agent monthly returns using vectorised operations."""
    results: dict[str, NDArray[Any]] = {}
    streams = (r_beta, r_H, r_E, r_M, f_int, f_ext_pa, f_act_ext)
    for agent in agents:
        alpha, financing = _resolve_streams(agent, *streams)
        results[agent.p.name] = agent.monthly_returns(r_beta, alpha, financing)

    return results
