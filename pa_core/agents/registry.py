from __future__ import annotations
from typing import Iterable, List

from . import (
    Agent,
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    ActiveExtensionAgent,
    InternalBetaAgent,
    InternalPAAgent,
)


_AGENT_MAP = {
    "Base": BaseAgent,
    "ExternalPA": ExternalPAAgent,
    "ActiveExt": ActiveExtensionAgent,
    "InternalBeta": InternalBetaAgent,
    "InternalPA": InternalPAAgent,
}


def build_all(params_list: Iterable[AgentParams]) -> List[Agent]:
    """Instantiate agents from a list of AgentParams."""
    agents: List[Agent] = []
    for p in params_list:
        cls = _AGENT_MAP.get(p.name)
        if cls is None:
            raise KeyError(f"Unknown agent name: {p.name}")
        agents.append(cls(p))
    return agents

