from __future__ import annotations

from importlib.metadata import entry_points
from typing import Iterable, List, Type

from ..config import ModelConfig
from ..validators import calculate_margin_requirement
from .active_ext import ActiveExtensionAgent
from .base import BaseAgent
from .external_pa import ExternalPAAgent
from .internal_beta import InternalBetaAgent
from .internal_pa import InternalPAAgent
from .types import Agent, AgentParams

_AGENT_MAP: dict[str, Type[Agent]] = {
    "Base": BaseAgent,
    "ExternalPA": ExternalPAAgent,
    "ActiveExt": ActiveExtensionAgent,
    "InternalBeta": InternalBetaAgent,
    "InternalPA": InternalPAAgent,
}


def register_agent(name: str, cls: Type[Agent]) -> None:
    """Register ``cls`` under ``name`` for agent construction."""
    if name in _AGENT_MAP:
        raise KeyError(f"Agent already registered: {name}")
    _AGENT_MAP[name] = cls


def load_plugins() -> None:
    """Load third-party agent plugins via entry points."""
    for ep in entry_points(group="pa_core.agents"):
        register_agent(ep.name, ep.load())


# Load any agent plugins on import
load_plugins()


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

    params: list[AgentParams] = []
    for agent in cfg.agents:
        if isinstance(agent, dict):
            name = agent["name"]
            capital = agent["capital"]
            beta_share = agent["beta_share"]
            alpha_share = agent["alpha_share"]
            extra = agent.get("extra", {})
        else:
            name = agent.name
            capital = agent.capital
            beta_share = agent.beta_share
            alpha_share = agent.alpha_share
            extra = agent.extra
        # Thread the (monthly) alpha mean and transfer-coefficient/cost controls into
        # the sleeves whose alpha lever supports diminishing returns (issue #1924).
        # cfg.mu_E / cfg.mu_M are already normalised to monthly by ModelConfig.
        if name == "ActiveExt":
            extra = {
                **extra,
                "alpha_mu": float(cfg.mu_E),
                "tc_decay": float(cfg.active_share_tc_decay),
                "cost_per_share": float(cfg.active_ext_cost_per_share),
            }
        elif name == "ExternalPA":
            extra = {
                **extra,
                "alpha_mu": float(cfg.mu_M),
                "tc_decay": float(cfg.theta_tc_decay),
                "cost_per_share": float(cfg.ext_pa_cost_per_share),
            }
        params.append(
            AgentParams(
                name,
                capital,
                beta_share,
                alpha_share,
                extra,
            )
        )

    margin_requirement = calculate_margin_requirement(
        reference_sigma=cfg.reference_sigma,
        volatility_multiple=cfg.volatility_multiple,
        total_capital=total_cap,
        financing_model=cfg.financing_model,
        schedule_path=cfg.financing_schedule_path,
        term_months=cfg.financing_term_months,
    )
    # InternalBeta here represents margin-backed capital, not attribution leftovers
    # (see docs/UserGuide.md "Sleeve Attribution Methodology" for InternalBeta vs UnexplainedBeta).
    if margin_requirement > 0 and not any(p.name == "InternalBeta" for p in params):
        params.append(
            AgentParams(
                "InternalBeta",
                margin_requirement,
                margin_requirement / total_cap,
                0.0,
                {},
            )
        )

    return build_all(params)
