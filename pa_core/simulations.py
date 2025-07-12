"""Legacy wrapper module for simulation utilities."""

from __future__ import annotations

from typing import Any, Iterable

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
    for agent in agents:
        if isinstance(agent, BaseAgent):
            alpha = r_H
            financing = f_int
        elif isinstance(agent, ExternalPAAgent):
            alpha = r_M
            financing = f_ext_pa
        elif isinstance(agent, ActiveExtensionAgent):
            alpha = r_E
            financing = f_act_ext
        elif isinstance(agent, InternalBetaAgent):
            alpha = r_H
            financing = f_int
        elif isinstance(agent, InternalPAAgent):
            alpha = r_H
            financing = np.zeros_like(r_beta)
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported agent type: {type(agent)}")

        results[agent.p.name] = agent.monthly_returns(r_beta, alpha, financing)

    return results
