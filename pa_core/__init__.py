"""Core portable alpha utilities."""

try:  # pragma: no cover - optional dependency
    from . import viz
except Exception:  # pragma: no cover - viz may require heavy deps
    viz = None

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
try:  # pragma: no cover - optional heavy deps
    from .reporting import export_to_excel, print_summary
    from .reporting.sweep_excel import export_sweep_results
except Exception:  # pragma: no cover
    export_to_excel = print_summary = export_sweep_results = None
from .run_flags import RunFlags
from .orchestrator import SimulatorOrchestrator
from .sim import (
    draw_financing_series,
    draw_joint_returns,
    prepare_mc_universe,
    simulate_alpha_streams,
    simulate_financing,
)
from .sim.covariance import build_cov_matrix, build_cov_matrix_with_validation
from .sim.metrics import (
    annualised_return,
    annualised_vol,
    compound,
    shortfall_probability,
    summary_table,
    tracking_error,
    value_at_risk,
)
from .sweep import run_parameter_sweep, run_parameter_sweep_cached, sweep_results_to_dataframe
from .stress import STRESS_PRESETS, apply_stress_preset
from .presets import AlphaPreset, PresetLibrary
from .validators import (
    ValidationResult,
    PSDProjectionInfo,
    validate_correlations,
    validate_covariance_matrix_psd,
    validate_capital_allocation,
    validate_simulation_parameters,
    calculate_margin_requirement,
    load_margin_schedule,
    format_validation_messages,
)

__all__ = [
    "load_index_returns",
    "simulate_financing",
    "prepare_mc_universe",
    "build_cov_matrix",
    "build_cov_matrix_with_validation",
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
    "run_parameter_sweep_cached",
    "sweep_results_to_dataframe",
    "apply_stress_preset",
    "STRESS_PRESETS",
    "AlphaPreset",
    "PresetLibrary",
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
    # Validation functions
    "ValidationResult",
    "PSDProjectionInfo", 
    "validate_correlations",
    "validate_covariance_matrix_psd",
    "validate_capital_allocation",
    "validate_simulation_parameters",
    "calculate_margin_requirement",
    "load_margin_schedule",
    "format_validation_messages",
]
