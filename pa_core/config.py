from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


class ConfigError(ValueError):
    """Invalid configuration."""


__all__ = ["ModelConfig", "load_config", "ConfigError"]


class ModelConfig(BaseModel):
    """Validated simulation parameters."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    N_SIMULATIONS: int = Field(gt=0, alias="N_SIMULATIONS")
    N_MONTHS: int = Field(gt=0, alias="N_MONTHS")

    external_pa_capital: float = 0.0
    active_ext_capital: float = 0.0
    internal_pa_capital: float = 0.0
    total_fund_capital: float = 1000.0

    w_beta_H: float = 0.5
    w_alpha_H: float = 0.5
    theta_extpa: float = 0.5
    active_share: float = 0.5

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

    # Parameter sweep options
    analysis_mode: str = "returns"

    max_external_combined_pct: float = 30.0
    external_step_size_pct: float = 5.0

    in_house_return_min_pct: float = 2.0
    in_house_return_max_pct: float = 6.0
    in_house_return_step_pct: float = 2.0
    in_house_vol_min_pct: float = 1.0
    in_house_vol_max_pct: float = 3.0
    in_house_vol_step_pct: float = 1.0
    alpha_ext_return_min_pct: float = 1.0
    alpha_ext_return_max_pct: float = 5.0
    alpha_ext_return_step_pct: float = 2.0
    alpha_ext_vol_min_pct: float = 2.0
    alpha_ext_vol_max_pct: float = 4.0
    alpha_ext_vol_step_pct: float = 1.0

    external_pa_alpha_min_pct: float = 25.0
    external_pa_alpha_max_pct: float = 75.0
    external_pa_alpha_step_pct: float = 5.0
    active_share_min_pct: float = 20.0
    active_share_max_pct: float = 100.0
    active_share_step_pct: float = 5.0

    sd_multiple_min: float = 2.0
    sd_multiple_max: float = 4.0
    sd_multiple_step: float = 0.25

    # Margin calculation parameters
    reference_sigma: float = 0.01  # Monthly volatility for margin calculation
    volatility_multiple: float = 3.0  # Multiplier for margin requirement
    financing_model: str = "simple_proxy"  # or "schedule"
    financing_schedule_path: Optional[Path] = None
    financing_term_months: float = 1.0

    risk_metrics: List[str] = Field(
        default_factory=lambda: [
            "Return",
            "Risk",
            "ShortfallProb",
        ]
    )

    @model_validator(mode="after")
    def check_financing_model(self) -> "ModelConfig":
        valid = {"simple_proxy", "schedule"}
        if self.financing_model not in valid:
            raise ValueError(f"financing_model must be one of: {sorted(valid)}")
        if self.financing_model == "schedule" and self.financing_schedule_path is None:
            raise ValueError("financing_schedule_path required for schedule financing model")
        return self

    @model_validator(mode="after")
    def check_capital(self) -> "ModelConfig":
        from .validators import validate_capital_allocation
        
        cap_sum = (
            self.external_pa_capital
            + self.active_ext_capital
            + self.internal_pa_capital
        )
        if cap_sum > self.total_fund_capital:
            raise ValueError("Capital allocation exceeds total_fund_capital")
            
        # Enhanced capital validation with margin requirements
        validation_results = validate_capital_allocation(
            external_pa_capital=self.external_pa_capital,
            active_ext_capital=self.active_ext_capital,
            internal_pa_capital=self.internal_pa_capital,
            total_fund_capital=self.total_fund_capital,
            reference_sigma=self.reference_sigma,
            volatility_multiple=self.volatility_multiple,
            financing_model=self.financing_model,
            margin_schedule_path=self.financing_schedule_path,
            term_months=self.financing_term_months,
        )
        
        # Check for critical errors
        errors = [r for r in validation_results if not r.is_valid]
        if errors:
            error_messages = [r.message for r in errors]
            raise ValueError("; ".join(error_messages))
        
        if "ShortfallProb" not in self.risk_metrics:
            raise ConfigError("risk_metrics must include ShortfallProb")
        return self

    @model_validator(mode="after")
    def check_shares(self) -> "ModelConfig":
        tol = 1e-6
        for name, val in [("w_beta_H", self.w_beta_H), ("w_alpha_H", self.w_alpha_H)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if abs(self.w_beta_H + self.w_alpha_H - 1.0) > tol:
            raise ValueError("w_beta_H and w_alpha_H must sum to 1")
        for name, val in [("theta_extpa", self.theta_extpa), ("active_share", self.active_share)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def check_analysis_mode(self) -> "ModelConfig":
        valid_modes = ["capital", "returns", "alpha_shares", "vol_mult"]
        if self.analysis_mode not in valid_modes:
            raise ValueError(f"analysis_mode must be one of: {valid_modes}")
        return self

    @model_validator(mode="after") 
    def check_simulation_params(self) -> "ModelConfig":
        from .validators import validate_simulation_parameters
        
        # Collect step sizes for validation
        step_sizes = {
            'external_step_size_pct': self.external_step_size_pct,
            'in_house_return_step_pct': self.in_house_return_step_pct,
            'in_house_vol_step_pct': self.in_house_vol_step_pct,
            'alpha_ext_return_step_pct': self.alpha_ext_return_step_pct,
            'alpha_ext_vol_step_pct': self.alpha_ext_vol_step_pct,
            'external_pa_alpha_step_pct': self.external_pa_alpha_step_pct,
            'active_share_step_pct': self.active_share_step_pct,
            'sd_multiple_step': self.sd_multiple_step,
        }
        
        validation_results = validate_simulation_parameters(
            n_simulations=self.N_SIMULATIONS,
            step_sizes=step_sizes
        )
        
        # Only raise errors for critical validation failures
        errors = [r for r in validation_results if not r.is_valid]
        if errors:
            error_messages = [r.message for r in errors]
            raise ValueError("; ".join(error_messages))
        
        return self


def load_config(path: Union[str, Path, Dict[str, Any]]) -> ModelConfig:
    """Return ``ModelConfig`` parsed from YAML dictionary."""
    if isinstance(path, dict):
        data = path
    else:
        data = yaml.safe_load(Path(path).read_text())
    try:
        cfg = ModelConfig(**data)
    except ValidationError as e:  # pragma: no cover - explicit failure
        raise ValueError(str(e)) from e
    if "ShortfallProb" not in cfg.risk_metrics:
        raise ConfigError("risk_metrics must include ShortfallProb")
    return cfg
