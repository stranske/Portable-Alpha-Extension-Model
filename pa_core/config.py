from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    cast,
)

import numpy as np
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from .backend import BACKEND_UNAVAILABLE_DETAIL, SUPPORTED_BACKENDS
from .share_utils import SHARE_MAX, SHARE_MIN, SHARE_SUM_TOLERANCE, normalize_share


class ConfigError(ValueError):
    """Invalid configuration."""


MONTHS_PER_YEAR = 12
CANONICAL_RETURN_UNIT: Literal["monthly"] = "monthly"
DEFAULT_RETURN_UNIT: Literal["annual", "monthly"] = "annual"
DEFAULT_MEAN_CONVERSION: Literal["simple", "geometric"] = "simple"


def annual_mean_to_monthly(
    value: float, *, method: Literal["simple", "geometric"] = DEFAULT_MEAN_CONVERSION
) -> float:
    """Convert an annual mean return to a monthly mean."""
    if method == "geometric":
        return float((1.0 + value) ** (1.0 / MONTHS_PER_YEAR) - 1.0)
    return float(value / MONTHS_PER_YEAR)


def annual_vol_to_monthly(value: float) -> float:
    """Convert an annual volatility to a monthly volatility."""
    return value / math.sqrt(MONTHS_PER_YEAR)


def annual_cov_to_monthly(cov: np.ndarray) -> np.ndarray:
    """Convert an annual covariance matrix to monthly units."""
    return np.asarray(cov, dtype=float) / MONTHS_PER_YEAR


__all__ = [
    "AgentConfig",
    "ModelConfig",
    "load_config",
    "ConfigError",
    "get_field_mappings",
    "normalize_share",
    "MONTHS_PER_YEAR",
    "CANONICAL_RETURN_UNIT",
    "DEFAULT_MEAN_CONVERSION",
    "annual_mean_to_monthly",
    "annual_vol_to_monthly",
    "annual_cov_to_monthly",
]

logger = logging.getLogger(__name__)


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


class AgentConfig(BaseModel):
    name: str
    capital: float
    beta_share: float
    alpha_share: float
    extra: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Validated simulation parameters for the portable-alpha model.

    Use ``ModelConfig`` for run settings, capital allocation, and sweep ranges.
    Use :class:`pa_core.schema.Scenario` for index/asset inputs, correlations,
    and sleeve definitions. They intentionally serve different roles: this
    class controls how simulations run, while ``Scenario`` supplies the market
    data and portfolio structure. Pair with
    :func:`pa_core.schema.load_scenario` when running a full simulation that
    needs both run settings and market data.
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    _RETURN_UNIT_EXPLICIT_KEYS: ClassVar[dict[str, dict[str, tuple[str, ...]]]] = {
        "mu_H": {
            "annual": ("mu_H_annual", "In-House annual return (%)"),
            "monthly": ("mu_H_monthly",),
        },
        "sigma_H": {
            "annual": ("sigma_H_annual", "In-House annual vol (%)"),
            "monthly": ("sigma_H_monthly",),
        },
        "mu_E": {
            "annual": ("mu_E_annual", "Alpha-Extension annual return (%)"),
            "monthly": ("mu_E_monthly",),
        },
        "sigma_E": {
            "annual": ("sigma_E_annual", "Alpha-Extension annual vol (%)"),
            "monthly": ("sigma_E_monthly",),
        },
        "mu_M": {
            "annual": ("mu_M_annual", "External annual return (%)"),
            "monthly": ("mu_M_monthly",),
        },
        "sigma_M": {
            "annual": ("sigma_M_annual", "External annual vol (%)"),
            "monthly": ("sigma_M_monthly",),
        },
    }

    backend: str = Field(default="numpy")
    N_SIMULATIONS: int = Field(gt=0, alias="Number of simulations")
    N_MONTHS: int = Field(gt=0, alias="Number of months")
    return_unit: Literal["annual", "monthly"] = Field(
        default=DEFAULT_RETURN_UNIT,
        alias="return_unit",
        description="Input unit for return means/volatilities.",
    )
    return_unit_input: Literal["annual", "monthly"] = Field(
        default=DEFAULT_RETURN_UNIT,
        exclude=True,
        description="Original return unit provided before normalization.",
    )

    return_distribution: str = Field(default="normal", alias="Return distribution")
    return_t_df: float = Field(default=5.0, alias="Student-t df")
    return_copula: str = Field(default="gaussian", alias="Return copula")
    return_distribution_idx: Optional[str] = Field(default=None, alias="Index return distribution")
    return_distribution_H: Optional[str] = Field(default=None, alias="In-House return distribution")
    return_distribution_E: Optional[str] = Field(
        default=None, alias="Alpha-Extension return distribution"
    )
    return_distribution_M: Optional[str] = Field(
        default=None, alias="External PA return distribution"
    )

    external_pa_capital: float = Field(default=0.0, alias="External PA capital (mm)")
    active_ext_capital: float = Field(default=0.0, alias="Active Extension capital (mm)")
    internal_pa_capital: float = Field(default=0.0, alias="Internal PA capital (mm)")
    total_fund_capital: float = Field(default=1000.0, alias="Total fund capital (mm)")
    agents: List[AgentConfig] = Field(default_factory=list)

    w_beta_H: float = Field(default=0.5, alias="In-House beta share")
    w_alpha_H: float = Field(default=0.5, alias="In-House alpha share")
    theta_extpa: float = Field(default=0.5, alias="External PA alpha fraction")
    active_share: float = Field(
        default=0.5,
        alias="Active share (%)",
        validation_alias=AliasChoices("Active share (%)", "Active share"),
        description="Active share fraction (0..1)",
    )

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

    correlation_repair_mode: Literal["error", "warn_fix"] = "warn_fix"
    correlation_repair_shrinkage: float = Field(default=0.0)

    covariance_shrinkage: Literal["none", "ledoit_wolf"] = "none"
    vol_regime: Literal["single", "two_state"] = "single"
    vol_regime_window: int = 12

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
    ext_pa_spike_prob: float = Field(default=0.0, alias="External PA monthly spike prob")
    ext_pa_spike_factor: float = Field(default=0.0, alias="External PA spike multiplier")

    act_ext_financing_mean_month: float = Field(
        default=0.0, alias="Active Ext financing mean (monthly %)"
    )
    act_ext_financing_sigma_month: float = Field(
        default=0.0, alias="Active Ext financing vol (monthly %)"
    )
    act_ext_spike_prob: float = Field(default=0.0, alias="Active Ext monthly spike prob")
    act_ext_spike_factor: float = Field(default=0.0, alias="Active Ext spike multiplier")
    financing_mode: Literal["broadcast", "per_path"] = Field(
        ...,
        description=(
            "Financing draw mode. broadcast reuses one financing vector across "
            "all simulations; per_path draws independent financing paths per scenario."
        ),
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
            "terminal_ShortfallProb",
        ],
        alias="risk_metrics",
    )

    debug_transform_order: bool = Field(
        default=False,
        exclude=True,
        description="Log transform execution order when true.",
    )

    # Transform/validation pipeline (explicit order):
    # 1) parse raw input
    # 2) normalize units (including share normalization)
    # 3) compile derived fields
    # 4) validate invariants (after validators below).
    @staticmethod
    def _trace_transform(data_or_self: Any, step: str) -> None:
        if isinstance(data_or_self, dict):
            debug = bool(data_or_self.get("debug_transform_order", False))
        else:
            debug = bool(getattr(data_or_self, "debug_transform_order", False))
        if debug:
            logger.info("ModelConfig transform: %s", step)

    @classmethod
    def _transform_pipeline(cls) -> tuple[tuple[str, Callable[[Any], Any]], ...]:
        """Return the ordered pre-validation transform pipeline."""
        return (
            ("normalize_return_units", cls.normalize_return_units),
            ("normalize_share_inputs", cls.normalize_share_inputs),
            ("compile_agent_config", cls.compile_agent_config),
        )

    @classmethod
    def _apply_explicit_return_units(
        cls, data: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, list[str]]:
        raw_keys = set(data)
        updated = dict(data)
        explicit_unit: str | None = None
        explicit_fields: set[str] = set()
        legacy_fields: list[str] = []

        for field, unit_keys in cls._RETURN_UNIT_EXPLICIT_KEYS.items():
            annual_keys = [key for key in unit_keys["annual"] if key in updated]
            monthly_keys = [key for key in unit_keys["monthly"] if key in updated]
            if annual_keys and monthly_keys:
                raise ValueError(
                    f"Ambiguous return unit for {field}: provide only annual or monthly values."
                )
            if len(annual_keys) > 1 or len(monthly_keys) > 1:
                raise ValueError(
                    f"Ambiguous return unit for {field}: multiple keys provided for the same unit."
                )
            if annual_keys or monthly_keys:
                unit = "annual" if annual_keys else "monthly"
                if explicit_unit and explicit_unit != unit:
                    raise ValueError(
                        "Conflicting return units provided across return inputs. "
                        "Use a single unit for mu/sigma inputs."
                    )
                explicit_unit = unit
                key = (annual_keys or monthly_keys)[0]
                if field in raw_keys:
                    raise ValueError(
                        f"Ambiguous return unit for {field}: do not mix {field} with {key}."
                    )
                updated[field] = updated[key]
                del updated[key]
                explicit_fields.add(field)
            if field in raw_keys and field not in explicit_fields:
                legacy_fields.append(field)

        return updated, explicit_unit, legacy_fields

    @classmethod
    def compile_agent_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cls._trace_transform(data, "compile_agent_config")

        convenience_keys = {
            "external_pa_capital",
            "active_ext_capital",
            "internal_pa_capital",
            "w_beta_H",
            "w_alpha_H",
            "theta_extpa",
            "active_share",
            "External PA capital (mm)",
            "Active Extension capital (mm)",
            "Internal PA capital (mm)",
            "In-House beta share",
            "In-House alpha share",
            "External PA alpha fraction",
            "Active share (%)",
            "Active share",
        }
        agents_provided = "agents" in data
        convenience_used = any(key in data for key in convenience_keys)
        if agents_provided and not convenience_used:
            raw_agents = data.get("agents") or []
            data["agents"] = cls._normalize_agents(raw_agents)
            return data

        def _get_value(field: str, aliases: tuple[str, ...] = ()) -> Any:
            for key in (field, *aliases):
                if key in data:
                    return data[key]
            field_info = cls.model_fields[field]
            if field_info.default_factory is not None:
                factory = cast(Callable[[], Any], field_info.default_factory)
                return factory()
            return field_info.default

        total_cap = float(_get_value("total_fund_capital", ("Total fund capital (mm)",)))
        w_beta = normalize_share(_get_value("w_beta_H", ("In-House beta share",)))
        w_alpha = normalize_share(_get_value("w_alpha_H", ("In-House alpha share",)))
        theta = normalize_share(_get_value("theta_extpa", ("External PA alpha fraction",)))
        active_share = normalize_share(
            _get_value("active_share", ("Active share (%)", "Active share"))
        )
        ext_cap = float(_get_value("external_pa_capital", ("External PA capital (mm)",)))
        act_cap = float(_get_value("active_ext_capital", ("Active Extension capital (mm)",)))
        int_cap = float(_get_value("internal_pa_capital", ("Internal PA capital (mm)",)))

        w_beta = 0.0 if w_beta is None else float(w_beta)
        w_alpha = 0.0 if w_alpha is None else float(w_alpha)
        theta = 0.0 if theta is None else float(theta)
        active_share = 0.0 if active_share is None else float(active_share)

        compiled: list[dict[str, Any]] = [
            {
                "name": "Base",
                "capital": total_cap,
                "beta_share": w_beta,
                "alpha_share": w_alpha,
                "extra": {},
            }
        ]

        if ext_cap > 0:
            compiled.append(
                {
                    "name": "ExternalPA",
                    "capital": ext_cap,
                    "beta_share": ext_cap / total_cap,
                    "alpha_share": 0.0,
                    "extra": {"theta_extpa": theta},
                }
            )

        if act_cap > 0:
            compiled.append(
                {
                    "name": "ActiveExt",
                    "capital": act_cap,
                    "beta_share": act_cap / total_cap,
                    "alpha_share": 0.0,
                    "extra": {"active_share": active_share},
                }
            )

        if int_cap > 0:
            compiled.append(
                {
                    "name": "InternalPA",
                    "capital": int_cap,
                    "beta_share": 0.0,
                    "alpha_share": int_cap / total_cap,
                    "extra": {},
                }
            )

        existing_agents = data.get("agents") or []
        if agents_provided:
            normalized_existing = cls._normalize_agents(existing_agents)
            existing_names = {agent["name"] for agent in normalized_existing}
            compiled = [
                agent for agent in compiled if agent["name"] not in existing_names
            ] + normalized_existing
        normalized = cls._normalize_agents(compiled)
        data["agents"] = normalized
        return data

    @classmethod
    def normalize_share_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cls._trace_transform(data, "normalize_share_inputs")
        share_fields = {
            "w_beta_H": ("In-House beta share",),
            "w_alpha_H": ("In-House alpha share",),
            "active_share": ("Active share (%)", "Active share"),
            "theta_extpa": ("External PA alpha fraction",),
        }
        for field, aliases in share_fields.items():
            for key in (field, *aliases):
                if key in data:
                    data[key] = normalize_share(data[key])
        return data

    @classmethod
    def normalize_return_units(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cls._trace_transform(data, "normalize_return_units")
        updated, explicit_unit, legacy_fields = cls._apply_explicit_return_units(data)
        unit = explicit_unit or updated.get("return_unit", DEFAULT_RETURN_UNIT)
        if explicit_unit and "return_unit" in updated and updated["return_unit"] != explicit_unit:
            raise ValueError(
                "return_unit conflicts with explicitly unit-tagged return inputs. "
                "Align return_unit with the explicit mu/sigma unit or remove return_unit."
            )
        updated["return_unit_input"] = unit
        if "return_unit" not in updated:
            updated["return_unit"] = unit
        if legacy_fields:
            warning_fields = ", ".join(sorted(legacy_fields))
            warnings.warn(
                "Ambiguous return inputs detected for legacy fields: "
                f"{warning_fields}. Use explicit keys like mu_H_annual or mu_H_monthly. "
                "Legacy field names will be deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        if unit == CANONICAL_RETURN_UNIT:
            return updated
        if unit != DEFAULT_RETURN_UNIT:
            return updated

        # Allow values to be supplied via field names or aliases.
        def _get_value(field: str) -> tuple[Any, list[str]]:
            field_info = cls.model_fields[field]
            candidates = [field]
            alias = getattr(field_info, "alias", None)
            if alias:
                candidates.append(alias)
            for key in candidates:
                if key in updated:
                    return updated[key], candidates
            return field_info.default, candidates

        for field in ("mu_H", "mu_E", "mu_M"):
            value, keys = _get_value(field)
            converted = annual_mean_to_monthly(value)
            for key in keys:
                updated[key] = converted
        for field in ("sigma_H", "sigma_E", "sigma_M"):
            value, keys = _get_value(field)
            converted = annual_vol_to_monthly(value)
            for key in keys:
                updated[key] = converted
        updated["return_unit"] = CANONICAL_RETURN_UNIT
        return updated

    @model_validator(mode="before")
    @classmethod
    def apply_transform_pipeline(cls, data: Any) -> Any:
        """Apply transforms in order: parse -> normalize units (incl. shares) -> compile -> validate."""
        if not isinstance(data, dict):
            return data
        cls._trace_transform(data, "parse_raw")
        for _, transform in cls._transform_pipeline():
            data = transform(data)
        return data

    @staticmethod
    def _normalize_agents(raw_agents: Any) -> list[dict[str, Any]]:
        if raw_agents is None:
            return []
        if not isinstance(raw_agents, list):
            raise ValueError("agents must be a list of mappings")
        normalized: list[dict[str, Any]] = []
        for idx, agent in enumerate(raw_agents):
            if isinstance(agent, AgentConfig):
                agent_data = agent.model_dump()
            elif isinstance(agent, Mapping):
                agent_data = dict(agent)
            else:
                raise ValueError(f"agents[{idx}] must be a mapping")
            missing = {"name", "capital", "beta_share", "alpha_share"} - agent_data.keys()
            if missing:
                raise ValueError(f"agents[{idx}] missing keys: {sorted(missing)}")
            extra = agent_data.get("extra") or {}
            if not isinstance(extra, dict):
                raise ValueError(f"agents[{idx}].extra must be a mapping")
            beta_share = normalize_share(agent_data["beta_share"])
            alpha_share = normalize_share(agent_data["alpha_share"])
            normalized.append(
                {
                    "name": str(agent_data["name"]),
                    "capital": float(agent_data["capital"]),
                    "beta_share": 0.0 if beta_share is None else float(beta_share),
                    "alpha_share": 0.0 if alpha_share is None else float(alpha_share),
                    "extra": extra,
                }
            )
        return normalized

    @model_validator(mode="after")
    def check_financing_model(self) -> "ModelConfig":
        self._trace_transform(self, "check_financing_model")
        valid = {"simple_proxy", "schedule"}
        if self.financing_model not in valid:
            raise ValueError(f"financing_model must be one of: {sorted(valid)}")
        if self.financing_model == "schedule" and self.financing_schedule_path is None:
            raise ValueError("financing_schedule_path required for schedule financing model")
        return self

    @model_validator(mode="after")
    def check_capital(self) -> "ModelConfig":
        self._trace_transform(self, "check_capital")
        from .validators import validate_capital_allocation

        cap_sum = self.external_pa_capital + self.active_ext_capital + self.internal_pa_capital
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

        if "terminal_ShortfallProb" not in self.risk_metrics:
            raise ConfigError("risk_metrics must include terminal_ShortfallProb")
        return self

    @model_validator(mode="after")
    def check_return_distribution(self) -> "ModelConfig":
        self._trace_transform(self, "check_return_distribution")
        valid_distributions = {"normal", "student_t"}
        valid_copulas = {"gaussian", "t"}
        if self.return_distribution not in valid_distributions:
            raise ValueError(f"return_distribution must be one of: {sorted(valid_distributions)}")
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
    def check_correlations(self) -> "ModelConfig":
        self._trace_transform(self, "check_correlations")
        from .validators import validate_correlations

        correlation_map = {
            "rho_idx_H": self.rho_idx_H,
            "rho_idx_E": self.rho_idx_E,
            "rho_idx_M": self.rho_idx_M,
            "rho_H_E": self.rho_H_E,
            "rho_H_M": self.rho_H_M,
            "rho_E_M": self.rho_E_M,
        }
        validation_results = validate_correlations(correlation_map)
        errors = [r for r in validation_results if not r.is_valid]
        if errors:
            error_messages = [r.message for r in errors]
            raise ValueError("; ".join(error_messages))
        return self

    @model_validator(mode="after")
    def check_correlation_repairs(self) -> "ModelConfig":
        self._trace_transform(self, "check_correlation_repairs")
        if not 0.0 <= self.correlation_repair_shrinkage <= 1.0:
            raise ValueError("correlation_repair_shrinkage must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def check_shares(self) -> "ModelConfig":
        self._trace_transform(self, "check_shares")
        for name, val in [("w_beta_H", self.w_beta_H), ("w_alpha_H", self.w_alpha_H)]:
            if not SHARE_MIN <= val <= SHARE_MAX:
                raise ValueError(f"{name} must be between 0 and 1")
        if abs(self.w_beta_H + self.w_alpha_H - 1.0) > SHARE_SUM_TOLERANCE:
            raise ValueError("w_beta_H and w_alpha_H must sum to 1")
        for name, val in [
            ("theta_extpa", self.theta_extpa),
            ("active_share", self.active_share),
        ]:
            if not SHARE_MIN <= val <= SHARE_MAX:
                raise ValueError(f"{name} must be between 0 and 1")
        return self

    @model_validator(mode="after")
    def check_agents(self) -> "ModelConfig":
        errors: list[str] = []
        names = [agent.name for agent in self.agents]
        if not names:
            errors.append("agents must include at least one agent")
        name_to_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(names):
            name_to_indices.setdefault(name, []).append(idx)
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            duplicate_detail = ", ".join(
                f"{name} (indices {name_to_indices[name]})" for name in duplicate_names
            )
            errors.append("agent names must be unique; duplicates found: " + duplicate_detail)
        benchmark_count = sum(1 for name in names if name == "Base")
        if benchmark_count != 1:
            base_indices = name_to_indices.get("Base", [])
            extra_hint = ""
            if benchmark_count == 0:
                extra_hint = " Add an agent with name 'Base' to the agents list."
                if names:
                    existing_names = ", ".join(sorted(set(names)))
                    extra_hint += f" Existing agents: {existing_names}."
            else:
                extra_hint = f" Found Base at indices {base_indices}; remove duplicates."
            errors.append(
                f"exactly one benchmark agent named 'Base' is required; found {benchmark_count}."
                f"{extra_hint}"
            )

        for idx, agent in enumerate(self.agents):
            if agent.capital < 0:
                errors.append(
                    f"agents[{idx}] ({agent.name}) capital must be >= 0; got {agent.capital}"
                )
            if not SHARE_MIN <= agent.beta_share <= SHARE_MAX:
                errors.append(
                    f"agents[{idx}] ({agent.name}) beta_share must be between 0 and 1; got {agent.beta_share}"
                )
            if not SHARE_MIN <= agent.alpha_share <= SHARE_MAX:
                errors.append(
                    f"agents[{idx}] ({agent.name}) alpha_share must be between 0 and 1; got {agent.alpha_share}"
                )
            if agent.beta_share + agent.alpha_share > SHARE_MAX + SHARE_SUM_TOLERANCE:
                errors.append(
                    f"agents[{idx}] ({agent.name}) beta_share + alpha_share must be <= 1; "
                    f"got {agent.beta_share + agent.alpha_share}. "
                    "Reduce beta_share or alpha_share so the total is <= 1."
                )

        if self.agents:
            total_agent_capital = sum(
                agent.capital for agent in self.agents if agent.name != "Base"
            )
            if total_agent_capital > self.total_fund_capital + SHARE_SUM_TOLERANCE:
                errors.append(
                    "sum(non-benchmark agent capital) must be <= total_fund_capital; "
                    f"got {total_agent_capital} > {self.total_fund_capital}. "
                    "Benchmark capital is treated as the total fund capital."
                )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def check_analysis_mode(self) -> "ModelConfig":
        self._trace_transform(self, "check_analysis_mode")
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
    def check_vol_regime_window(self) -> "ModelConfig":
        self._trace_transform(self, "check_vol_regime_window")
        if self.vol_regime == "two_state" and self.vol_regime_window <= 1:
            raise ValueError("vol_regime_window must be > 1 for two_state regime")
        return self

    @model_validator(mode="after")
    def check_backend(self) -> "ModelConfig":
        self._trace_transform(self, "check_backend")
        valid_backends = list(SUPPORTED_BACKENDS)
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of: {valid_backends} ({BACKEND_UNAVAILABLE_DETAIL})"
            )
        return self

    @model_validator(mode="after")
    def check_simulation_params(self) -> "ModelConfig":
        self._trace_transform(self, "check_simulation_params")
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

    See :class:`pa_core.schema.Scenario` for market data inputs that pair with
    the simulation parameters defined here.

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
        if isinstance(data, dict):
            schedule_path = data.get("financing_schedule_path")
            if schedule_path:
                schedule_path = Path(schedule_path)
                if not schedule_path.is_absolute():
                    data["financing_schedule_path"] = p.parent.joinpath(schedule_path).resolve()
    try:
        cfg = ModelConfig(**data)
    except ValidationError as e:  # pragma: no cover - explicit failure
        raise ValueError(str(e)) from e
    if "terminal_ShortfallProb" not in cfg.risk_metrics:
        raise ConfigError("risk_metrics must include terminal_ShortfallProb")
    return cfg
