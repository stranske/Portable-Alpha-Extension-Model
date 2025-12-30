"""Stress scenario presets for Portable Alpha model.

Provides predefined overrides for common stress tests such as
liquidity squeezes or volatility regime shifts.  Presets return a new
``ModelConfig`` instance with the modifications applied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from .config import ModelConfig
from .validators import VOLATILITY_STRESS_MULTIPLIER

Preset = Mapping[str, float | Callable[[ModelConfig], float]]

STRESS_PRESET_LABELS: dict[str, str] = {
    "liquidity_squeeze": "Liquidity squeeze",
    "correlation_breakdown": "Correlation breakdown",
    "2008_vol_regime": "2008-like vol regime",
    "2020_gap_day": "2020 gap day",
}


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
        "sigma_H": lambda cfg: cfg.sigma_H * VOLATILITY_STRESS_MULTIPLIER,
        "sigma_E": lambda cfg: cfg.sigma_E * VOLATILITY_STRESS_MULTIPLIER,
        "sigma_M": lambda cfg: cfg.sigma_M * VOLATILITY_STRESS_MULTIPLIER,
    },
    "2020_gap_day": {
        "mu_H": -0.20,
        "mu_E": -0.25,
        "mu_M": -0.15,
    },
}


def _normalize_preset_name(name: str) -> str:
    normalized = name.strip()
    if normalized in STRESS_PRESETS:
        return normalized
    normalized_lower = normalized.lower()
    for key, label in STRESS_PRESET_LABELS.items():
        if normalized_lower == label.lower():
            return key
    underscored = normalized_lower.replace("-", "_").replace(" ", "_")
    if underscored in STRESS_PRESETS:
        return underscored
    raise KeyError(f"Unknown stress preset: {name}")


def apply_stress_preset(cfg: ModelConfig, name: str) -> ModelConfig:
    """Return a copy of ``cfg`` with stress preset ``name`` applied.

    Parameters
    ----------
    cfg:
        Base configuration.
    name:
        Key in :data:`STRESS_PRESETS`.
    """
    preset_key = _normalize_preset_name(name)
    preset = STRESS_PRESETS[preset_key]
    updates = {k: (v(cfg) if callable(v) else v) for k, v in preset.items()}
    return cast(ModelConfig, cfg.model_copy(update=updates))


__all__ = ["STRESS_PRESETS", "STRESS_PRESET_LABELS", "apply_stress_preset"]
