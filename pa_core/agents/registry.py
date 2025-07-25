from __future__ import annotations

from typing import Iterable, List

from ..config import ModelConfig
from .active_ext import ActiveExtensionAgent
from .base import BaseAgent
from .external_pa import ExternalPAAgent
from .internal_beta import InternalBetaAgent
from .internal_pa import InternalPAAgent
from .types import Agent, AgentParams

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


def build_from_config(cfg: ModelConfig) -> List[Agent]:
    """Instantiate agents based on ``ModelConfig`` values."""
    total_cap = cfg.total_fund_capital

    params: list[AgentParams] = [
        AgentParams("Base", total_cap, cfg.w_beta_H, cfg.w_alpha_H, {})
    ]

    if cfg.external_pa_capital > 0:
        params.append(
            AgentParams(
                "ExternalPA",
                cfg.external_pa_capital,
                cfg.external_pa_capital / total_cap,
                0.0,
                {"theta_extpa": cfg.theta_extpa},
            )
        )

    if cfg.active_ext_capital > 0:
        params.append(
            AgentParams(
                "ActiveExt",
                cfg.active_ext_capital,
                cfg.active_ext_capital / total_cap,
                0.0,
                {"active_share": cfg.active_share},
            )
        )

    if cfg.internal_pa_capital > 0:
        params.append(
            AgentParams(
                "InternalPA",
                cfg.internal_pa_capital,
                0.0,
                cfg.internal_pa_capital / total_cap,
                {},
            )
        )

    leftover_beta = (
        total_cap
        - cfg.external_pa_capital
        - cfg.active_ext_capital
        - cfg.internal_pa_capital
    )
    if leftover_beta > 0:
        params.append(
            AgentParams(
                "InternalBeta", leftover_beta, leftover_beta / total_cap, 0.0, {}
            )
        )

    return build_all(params)
