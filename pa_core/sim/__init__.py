"""Vectorised Monte-Carlo helpers."""

from .paths import (
    simulate_financing,
    prepare_mc_universe,
    draw_joint_returns,
    draw_financing_series,
    simulate_alpha_streams,
)

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
]
