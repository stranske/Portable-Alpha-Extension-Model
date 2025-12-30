from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Invalid configuration."""


__all__ = ["ModelConfig", "load_config", "ConfigError", "get_field_mappings"]


def get_field_mappings(model_class: type[BaseModel] | None = None) -> Dict[str, str]:
    """
    Extract field mappings from a Pydantic model.

    Returns a dictionary mapping field aliases (human-readable names)
    to field names (snake_case), based on the model's field definitions.

    Args:
        model_class: Pydantic model class to extract mappings from.
                    Defaults to ModelConfig.

    Returns:
        Dictionary mapping alias -> field_name
    """
    if model_class is None:
        model_class = ModelConfig

    mappings = {}

    for field_name, field_info in model_class.model_fields.items():
        # Check if field has an alias
        if hasattr(field_info, "alias") and field_info.alias:
            alias = field_info.alias
            # Use the alias as the human-readable name
            mappings[alias] = field_name
        else:
            # For fields without aliases, use the field name as both key and value
            # This maintains backward compatibility
            mappings[field_name] = field_name

    return mappings


class ModelConfig(BaseModel):
    """Validated simulation parameters."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    backend: str = Field(default="numpy")
    N_SIMULATIONS: int = Field(gt=0, alias="Number of simulations")
    N_MONTHS: int = Field(gt=0, alias="Number of months")

    return_distribution: str = Field(default="normal", alias="Return distribution")
    return_t_df: float = Field(default=5.0, alias="Student-t df")
    return_copula: str = Field(default="gaussian", alias="Return copula")
    return_distribution_idx: Optional[str] = Field(
        default=None, alias="Index return distribution"
    )
    return_distribution_H: Optional[str] = Field(
        default=None, alias="In-House return distribution"
    )
    return_distribution_E: Optional[str] = Field(
        default=None, alias="Alpha-Extension return distribution"
    )
    return_distribution_M: Optional[str] = Field(
        default=None, alias="External PA return distribution"
    )

    external_pa_capital: float = Field(default=0.0, alias="External PA capital (mm)")
    active_ext_capital: float = Field(
        default=0.0, alias="Active Extension capital (mm)"
    )
    internal_pa_capital: float = Field(default=0.0, alias="Internal PA capital (mm)")
    total_fund_capital: float = Field(default=1000.0, alias="Total fund capital (mm)")

    w_beta_H: float = Field(default=0.5, alias="In-House beta share")
    w_alpha_H: float = Field(default=0.5, alias="In-House alpha share")
    theta_extpa: float = Field(default=0.5, alias="External PA alpha fraction")
    active_share: float = Field(default=0.5, alias="Active share (%)")

    mu_H: float = Field(default=0.04, alias="In-House annual return (%)")
    sigma_H: float = Field(default=0.01, alias="In-House annual vol (%)")
    mu_E: float = Field(default=0.05, alias="Alpha-Extension annual return (%)")
    sigma_E: float = Field(default=0.02, alias="Alpha-Extension annual vol (%)")
    mu_M: float = Field(default=0.03, alias="External annual return (%)")
    sigma_M: float = Field(default=0.02, alias="External annual vol (%)")

    rho_idx_H: float = Field(default=0.05, alias="Corr index–In-House")
    rho_idx_E: float = Field(default=0.0, alias="Corr index–Alpha-Extension")
    rho_idx_M: float = Field(default=0.0, alias="Corr index–External")
    rho_H_E: float = Field(default=0.10, alias="Corr In-House–Alpha-Extension")
    rho_H_M: float = Field(default=0.10, alias="Corr In-House–External")
    rho_E_M: float = Field(default=0.0, alias="Corr Alpha-Extension–External")

    internal_financing_mean_month: float = Field(
        default=0.0, alias="Internal financing mean (monthly %)"
    )
    internal_financing_sigma_month: float = Field(
        default=0.0, alias="Internal financing vol (monthly %)"
    )
    internal_spike_prob: float = Field(default=0.0, alias="Internal monthly spike prob")
    internal_spike_factor: float = Field(default=0.0, alias="Internal spike multiplier")

    ext_pa_financing_mean_month: float = Field(
        default=0.0, alias="External PA financing mean (monthly %)"
    )
    ext_pa_financing_sigma_month: float = Field(
        default=0.0, alias="External PA financing vol (monthly %)"
    )
    ext_pa_spike_prob: float = Field(
        default=0.0, alias="External PA monthly spike prob"
    )
    ext_pa_spike_factor: float = Field(
        default=0.0, alias="External PA spike multiplier"
    )

    act_ext_financing_mean_month: float = Field(
        default=0.0, alias="Active Ext financing mean (monthly %)"
    )
    act_ext_financing_sigma_month: float = Field(
        default=0.0, alias="Active Ext financing vol (monthly %)"
    )
    act_ext_spike_prob: float = Field(
        default=0.0, alias="Active Ext monthly spike prob"
    )
    act_ext_spike_factor: float = Field(
        default=0.0, alias="Active Ext spike multiplier"
    )

    # Parameter sweep options
    analysis_mode: str = Field(default="returns", alias="Analysis mode")

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
        ],
        alias="risk_metrics",
    )

    @model_validator(mode="after")
    def check_financing_model(self) -> "ModelConfig":
        valid = {"simple_proxy", "schedule"}
        if self.financing_model not in valid:
            raise ValueError(f"financing_model must be one of: {sorted(valid)}")
        if self.financing_model == "schedule" and self.financing_schedule_path is None:
            raise ValueError(
                "financing_schedule_path required for schedule financing model"
            )
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
    def check_return_distribution(self) -> "ModelConfig":
        valid_distributions = {"normal", "student_t"}
        valid_copulas = {"gaussian", "t"}
        if self.return_distribution not in valid_distributions:
            raise ValueError(
                f"return_distribution must be one of: {sorted(valid_distributions)}"
            )
        for dist in (
            self.return_distribution_idx,
            self.return_distribution_H,
            self.return_distribution_E,
            self.return_distribution_M,
        ):
            if dist is not None and dist not in valid_distributions:
                raise ValueError(
                    f"return_distribution must be one of: {sorted(valid_distributions)}"
                )
        if self.return_copula not in valid_copulas:
            raise ValueError(f"return_copula must be one of: {sorted(valid_copulas)}")
        resolved = (
            self.return_distribution_idx or self.return_distribution,
            self.return_distribution_H or self.return_distribution,
            self.return_distribution_E or self.return_distribution,
            self.return_distribution_M or self.return_distribution,
        )
        if all(dist == "normal" for dist in resolved) and self.return_copula != "gaussian":
            raise ValueError(
                "return_copula must be 'gaussian' when return_distribution is 'normal'"
            )
        if any(dist == "student_t" for dist in resolved) and self.return_t_df <= 2.0:
            raise ValueError("return_t_df must be greater than 2 for finite variance")
        return self

    @model_validator(mode="after")
    def check_shares(self) -> "ModelConfig":
        tol = 1e-6
        for name, val in [("w_beta_H", self.w_beta_H), ("w_alpha_H", self.w_alpha_H)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if abs(self.w_beta_H + self.w_alpha_H - 1.0) > tol:
            raise ValueError("w_beta_H and w_alpha_H must sum to 1")
        for name, val in [
            ("theta_extpa", self.theta_extpa),
            ("active_share", self.active_share),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def check_analysis_mode(self) -> "ModelConfig":
        valid_modes = [
            "capital",
            "returns",
            "alpha_shares",
            "vol_mult",
            "single_with_sensitivity",
        ]
        if self.analysis_mode not in valid_modes:
            raise ValueError(f"analysis_mode must be one of: {valid_modes}")
        return self

    @model_validator(mode="after")
    def check_backend(self) -> "ModelConfig":
        valid_backends = ["numpy", "cupy"]
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of: {valid_backends}")
        return self

    @model_validator(mode="after")
    def check_simulation_params(self) -> "ModelConfig":
        from .validators import validate_simulation_parameters

        # Collect step sizes for validation
        step_sizes = {
            "external_step_size_pct": self.external_step_size_pct,
            "in_house_return_step_pct": self.in_house_return_step_pct,
            "in_house_vol_step_pct": self.in_house_vol_step_pct,
            "alpha_ext_return_step_pct": self.alpha_ext_return_step_pct,
            "alpha_ext_vol_step_pct": self.alpha_ext_vol_step_pct,
            "external_pa_alpha_step_pct": self.external_pa_alpha_step_pct,
            "active_share_step_pct": self.active_share_step_pct,
            "sd_multiple_step": self.sd_multiple_step,
        }

        validation_results = validate_simulation_parameters(
            n_simulations=self.N_SIMULATIONS, step_sizes=step_sizes
        )

        # Only raise errors for critical validation failures
        errors = [r for r in validation_results if not r.is_valid]
        if errors:
            error_messages = [r.message for r in errors]
            raise ValueError("; ".join(error_messages))

        return self


def load_config(path: Union[str, Path, Dict[str, Any]]) -> ModelConfig:
    """Return ``ModelConfig`` parsed from YAML dictionary or file.

    Raises
    ------
    FileNotFoundError
        If ``path`` is a string/Path and the file does not exist.
    ConfigError
        If the YAML content cannot be parsed or mandatory fields are missing.
    """
    if isinstance(path, dict):
        data = path
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:  # pragma: no cover - user input
            raise ConfigError(f"Invalid YAML in config file {p}: {exc}") from exc
    try:
        cfg = ModelConfig(**data)
    except ValidationError as e:  # pragma: no cover - explicit failure
        raise ValueError(str(e)) from e
    if "ShortfallProb" not in cfg.risk_metrics:
        raise ConfigError("risk_metrics must include ShortfallProb")
    return cfg
