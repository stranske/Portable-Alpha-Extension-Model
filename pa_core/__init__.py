"""Core portable alpha utilities."""

from .io import (
    select_csv_file,
    load_parameters,
    get_num,
    build_range,
    build_range_int,
    load_index_returns,
)
from .simulations import (
    simulate_financing,
    prepare_mc_universe,
    build_cov_matrix,
    draw_joint_returns,
    draw_financing_series,
    simulate_alpha_streams,
)
from .reporting import export_to_excel
from .metrics import tracking_error, value_at_risk
from .agents import (
    Agent,
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    ActiveExtensionAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from .agents.registry import build_all as build_agents

__all__ = [
    "select_csv_file",
    "load_parameters",
    "get_num",
    "build_range",
    "build_range_int",
    "load_index_returns",
    "simulate_financing",
    "prepare_mc_universe",
    "build_cov_matrix",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
    "export_to_excel",
    "tracking_error",
    "value_at_risk",
    "Agent",
    "AgentParams",
    "BaseAgent",
    "ExternalPAAgent",
    "ActiveExtensionAgent",
    "InternalBetaAgent",
    "InternalPAAgent",
    "build_agents",
]
