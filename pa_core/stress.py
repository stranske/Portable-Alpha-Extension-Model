"""Stress scenario presets for Portable Alpha model.

Provides predefined overrides for common stress tests such as
liquidity squeezes or volatility regime shifts.  Presets return a new
``ModelConfig`` instance with the modifications applied.
"""

from __future__ import annotations

from typing import Callable, Mapping, cast

from .config import ModelConfig

Preset = Mapping[str, float | Callable[[ModelConfig], float]]


STRESS_PRESETS: dict[str, Preset] = {
    "liquidity_squeeze": {
        "internal_financing_sigma_month": 0.05,
        "ext_pa_financing_sigma_month": 0.05,
        "act_ext_financing_sigma_month": 0.05,
        "internal_spike_prob": 0.1,
        "internal_spike_factor": 5.0,
        "ext_pa_spike_prob": 0.1,
        "ext_pa_spike_factor": 5.0,
        "act_ext_spike_prob": 0.1,
        "act_ext_spike_factor": 5.0,
    },
    "correlation_breakdown": {
        "rho_idx_H": 0.95,
        "rho_idx_E": 0.95,
        "rho_idx_M": 0.95,
        "rho_H_E": 0.95,
        "rho_H_M": 0.95,
        "rho_E_M": 0.95,
    },
    "2008_vol_regime": {
        "sigma_H": lambda cfg: cfg.sigma_H * 3,
        "sigma_E": lambda cfg: cfg.sigma_E * 3,
        "sigma_M": lambda cfg: cfg.sigma_M * 3,
    },
    "2020_gap_day": {
        "mu_H": -0.20,
        "mu_E": -0.25,
        "mu_M": -0.15,
    },
}


def apply_stress_preset(cfg: ModelConfig, name: str) -> ModelConfig:
    """Return a copy of ``cfg`` with stress preset ``name`` applied.

    Parameters
    ----------
    cfg:
        Base configuration.
    name:
        Key in :data:`STRESS_PRESETS`.
    """
    if name not in STRESS_PRESETS:
        raise KeyError(f"Unknown stress preset: {name}")
    preset = STRESS_PRESETS[name]
    updates = {k: (v(cfg) if callable(v) else v) for k, v in preset.items()}
    return cast(ModelConfig, cfg.model_copy(update=updates))


__all__ = ["STRESS_PRESETS", "apply_stress_preset"]
