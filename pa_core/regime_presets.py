"""Preset regime configurations for stress scenario modeling."""

from __future__ import annotations

from typing import Callable

from .config import ModelConfig, RegimeConfig

RegimePreset = Callable[[ModelConfig], tuple[list[RegimeConfig], list[list[float]], str | None]]

REGIME_PRESET_LABELS: dict[str, str] = {
    "2008_crisis": "2008 crisis",
    "covid_shock": "COVID shock",
}


def _build_2008_crisis(cfg: ModelConfig) -> tuple[list[RegimeConfig], list[list[float]], str]:
    calm = RegimeConfig(name="calm")
    crisis = RegimeConfig(
        name="crisis",
        idx_sigma_multiplier=2.5,
        sigma_H=cfg.sigma_H * 2.5,
        sigma_E=cfg.sigma_E * 2.5,
        sigma_M=cfg.sigma_M * 2.5,
        rho_idx_H=0.85,
        rho_idx_E=0.85,
        rho_idx_M=0.85,
        rho_H_E=0.9,
        rho_H_M=0.9,
        rho_E_M=0.9,
    )
    transition = [
        [0.95, 0.05],
        [0.2, 0.8],
    ]
    return [calm, crisis], transition, "calm"


def _build_covid_shock(cfg: ModelConfig) -> tuple[list[RegimeConfig], list[list[float]], str]:
    calm = RegimeConfig(name="calm")
    shock = RegimeConfig(
        name="shock",
        idx_sigma_multiplier=3.0,
        sigma_H=cfg.sigma_H * 3.0,
        sigma_E=cfg.sigma_E * 3.0,
        sigma_M=cfg.sigma_M * 2.5,
        rho_idx_H=0.9,
        rho_idx_E=0.88,
        rho_idx_M=0.85,
        rho_H_E=0.88,
        rho_H_M=0.85,
        rho_E_M=0.82,
    )
    transition = [
        [0.97, 0.03],
        [0.6, 0.4],
    ]
    return [calm, shock], transition, "calm"


REGIME_PRESETS: dict[str, RegimePreset] = {
    "2008_crisis": _build_2008_crisis,
    "covid_shock": _build_covid_shock,
}


def _normalize_preset_name(name: str) -> str:
    normalized = name.strip()
    if normalized in REGIME_PRESETS:
        return normalized
    normalized_lower = normalized.lower()
    for key, label in REGIME_PRESET_LABELS.items():
        if normalized_lower == label.lower():
            return key
    underscored = normalized_lower.replace("-", "_").replace(" ", "_")
    if underscored in REGIME_PRESETS:
        return underscored
    raise KeyError(f"Unknown regime preset: {name}")


def apply_regime_preset(cfg: ModelConfig, name: str) -> ModelConfig:
    preset_key = _normalize_preset_name(name)
    regimes, transition, start = REGIME_PRESETS[preset_key](cfg)
    return cfg.model_copy(
        update={
            "regimes": regimes,
            "regime_transition": transition,
            "regime_start": start,
        }
    )


__all__ = ["REGIME_PRESETS", "REGIME_PRESET_LABELS", "apply_regime_preset"]
