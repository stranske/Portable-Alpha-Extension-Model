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
    RiskMetricsAgent,
    RiskMetrics,
)
from .agents.registry import build_all as build_agents
from .agents.registry import build_from_config
from .backend import get_backend, set_backend
from .config import ConfigError, ModelConfig, load_config
from .data import load_index_returns
from .random import spawn_agent_rngs, spawn_rngs
from .reporting import export_to_excel, print_summary
from .reporting.sweep_excel import export_sweep_results
from .run_flags import RunFlags
from .orchestrator import SimulatorOrchestrator
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
    shortfall_probability,
    summary_table,
    tracking_error,
    value_at_risk,
)
from .sweep import run_parameter_sweep

__all__ = [
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
    "export_sweep_results",
    "print_summary",
    "run_parameter_sweep",
    "tracking_error",
    "value_at_risk",
    "compound",
    "annualised_return",
    "annualised_vol",
    "summary_table",
    "shortfall_probability",
    "Agent",
    "AgentParams",
    "BaseAgent",
    "ExternalPAAgent",
    "ActiveExtensionAgent",
    "InternalBetaAgent",
    "InternalPAAgent",
    "RiskMetricsAgent",
    "RiskMetrics",
    "ModelConfig",
    "load_config",
    "ConfigError",
    "RunFlags",
    "build_agents",
    "build_from_config",
    "SimulatorOrchestrator",
    "viz",
]
