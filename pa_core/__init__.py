"""Core portable alpha utilities."""

from . import viz
from .agents import (
    ActiveExtensionAgent,
    Agent,
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from .agents.registry import build_all as build_agents
from .agents.registry import build_from_config
from .backend import get_backend, set_backend
from .config import ConfigError, ModelConfig, load_config
from .data import (
    build_range,
    build_range_int,
    get_num,
    load_index_returns,
    load_parameters,
    select_csv_file,
)
from .random import spawn_agent_rngs, spawn_rngs
from .reporting import export_to_excel, print_summary
from .run_flags import RunFlags
from .sim import (
    draw_financing_series,
    draw_joint_returns,
    prepare_mc_universe,
    simulate_alpha_streams,
    simulate_financing,
)
from .sim.covariance import build_cov_matrix
from .sim.metrics import (
    annualised_return,
    annualised_vol,
    compound,
    summary_table,
    tracking_error,
    value_at_risk,
)

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
    "ConfigError",
    "RunFlags",
    "build_agents",
    "build_from_config",
    "viz",
]
