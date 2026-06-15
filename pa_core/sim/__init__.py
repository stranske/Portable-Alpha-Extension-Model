"""Vectorised Monte-Carlo helpers."""

from .covariance import build_cov_matrix, build_generic_cov_matrix
from .financing import (
    broadcast_dispersion_warning,
    draw_financing_series,
    simulate_financing,
)
from .internal_pa_financing import resolve_internal_pa_financing_series
from .paths import (
    draw_joint_returns,
    draw_named_returns,
    draw_returns,
    map_sleeve_alpha_streams,
    prepare_mc_universe,
    prepare_return_shocks,
    simulate_alpha_streams,
)
from .regimes import (
    apply_regime_labels,
    build_regime_draw_params,
    resolve_regime_start,
    simulate_regime_paths,
)

# Backward-compatible alias for draw_financing_series
draw_financing = draw_financing_series

__all__ = [
    "simulate_financing",
    "broadcast_dispersion_warning",
    "prepare_mc_universe",
    "build_cov_matrix",
    "build_generic_cov_matrix",
    "prepare_return_shocks",
    "draw_named_returns",
    "draw_returns",
    "draw_financing",
    "draw_joint_returns",
    "draw_financing_series",
    "map_sleeve_alpha_streams",
    "resolve_internal_pa_financing_series",
    "simulate_alpha_streams",
    "apply_regime_labels",
    "build_regime_draw_params",
    "resolve_regime_start",
    "simulate_regime_paths",
]
