"""Core portable alpha utilities."""

from .data import (
    select_csv_file,
    load_parameters,
    get_num,
    build_range,
    build_range_int,
    load_index_returns,
)
from .sim import (
    simulate_financing,
    prepare_mc_universe,
    draw_joint_returns,
    draw_financing_series,
    simulate_alpha_streams,
)
from .sim.covariance import build_cov_matrix
from .random import spawn_rngs, spawn_agent_rngs
from .backend import set_backend, get_backend
from .reporting import export_to_excel, print_summary
from . import viz
from .sim.metrics import (
    tracking_error,
    value_at_risk,
    compound,
    annualised_return,
    annualised_vol,
    summary_table,
)
from .config import ModelConfig, load_config
from .agents import (
    Agent,
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    ActiveExtensionAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from .agents.registry import build_all as build_agents, build_from_config

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
    "spawn_rngs",
    "spawn_agent_rngs",
    "set_backend",
    "get_backend",
    "export_to_excel",
    "print_summary",
    "tracking_error",
    "value_at_risk",
    "compound",
    "annualised_return",
    "annualised_vol",
    "summary_table",
    "Agent",
    "AgentParams",
    "BaseAgent",
    "ExternalPAAgent",
    "ActiveExtensionAgent",
    "InternalBetaAgent",
    "InternalPAAgent",
    "ModelConfig",
    "load_config",
    "build_agents",
    "build_from_config",
    "viz",
]
