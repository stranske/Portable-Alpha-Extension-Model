from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

__all__ = ["ModelConfig", "load_config"]


class ModelConfig(BaseModel):
    """Validated simulation parameters."""

    N_SIMULATIONS: int = Field(gt=0, alias="N_SIMULATIONS")
    N_MONTHS: int = Field(gt=0, alias="N_MONTHS")

    external_pa_capital: float = 0.0
    active_ext_capital: float = 0.0
    internal_pa_capital: float = 0.0
    total_fund_capital: float = 1000.0

    mu_H: float = 0.04
    sigma_H: float = 0.01
    mu_E: float = 0.05
    sigma_E: float = 0.02
    mu_M: float = 0.03
    sigma_M: float = 0.02

    rho_idx_H: float = 0.05
    rho_idx_E: float = 0.0
    rho_idx_M: float = 0.0
    rho_H_E: float = 0.10
    rho_H_M: float = 0.10
    rho_E_M: float = 0.0

    internal_financing_mean_month: float = 0.0
    internal_financing_sigma_month: float = 0.0
    internal_spike_prob: float = 0.0
    internal_spike_factor: float = 0.0

    ext_pa_financing_mean_month: float = 0.0
    ext_pa_financing_sigma_month: float = 0.0
    ext_pa_spike_prob: float = 0.0
    ext_pa_spike_factor: float = 0.0

    act_ext_financing_mean_month: float = 0.0
    act_ext_financing_sigma_month: float = 0.0
    act_ext_spike_prob: float = 0.0
    act_ext_spike_factor: float = 0.0

    class Config:
        allow_population_by_field_name = True
        frozen = True

    @model_validator(mode="after")
    def check_capital(self) -> "ModelConfig":
        cap_sum = (
            self.external_pa_capital
            + self.active_ext_capital
            + self.internal_pa_capital
        )
        if cap_sum > self.total_fund_capital:
            raise ValueError(
                "Capital allocation exceeds total_fund_capital"
            )
        return self


def load_config(path: str | Path | dict[str, Any]) -> ModelConfig:
    """Return ``ModelConfig`` parsed from YAML/CSV dictionary."""
    if isinstance(path, dict):
        data = path
    else:
        data = yaml.safe_load(Path(path).read_text())
    try:
        return ModelConfig(**data)
    except ValidationError as e:  # pragma: no cover - explicit failure
        raise ValueError(str(e)) from e
