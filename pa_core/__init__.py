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
    draw_joint_returns,
    draw_financing_series,
    simulate_alpha_streams,
)
from .covariance import build_cov_matrix
from .random import spawn_rngs
from .reporting import export_to_excel
from .metrics import tracking_error, value_at_risk
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
    "spawn_rngs",
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
    "ModelConfig",
    "load_config",
    "build_agents",
]
