"""Vectorised Monte-Carlo helpers."""

from .covariance import build_cov_matrix
from .paths import (
    draw_financing_series,
    draw_joint_returns,
    prepare_mc_universe,
    prepare_return_shocks,
    simulate_alpha_streams,
    simulate_financing,
)
from .regimes import apply_regime_labels, build_regime_draw_params, resolve_regime_start, simulate_regime_paths

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "build_cov_matrix",
    "prepare_return_shocks",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
    "apply_regime_labels",
    "build_regime_draw_params",
    "resolve_regime_start",
    "simulate_regime_paths",
]
