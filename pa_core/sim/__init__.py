"""Vectorised Monte-Carlo helpers."""

from .paths import (
    simulate_financing,
    prepare_mc_universe,
    draw_joint_returns,
    draw_financing_series,
    simulate_alpha_streams,
)
from .covariance import build_cov_matrix

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "build_cov_matrix",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
]
