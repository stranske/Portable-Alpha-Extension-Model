"""Unit policy and conversion helpers for portable-alpha simulations.

Policy summary
--------------
- Configured return inputs (mu/sigma/cov) are normalized to monthly units.
  ``ModelConfig.return_unit`` reports the canonical unit after normalization,
  while ``return_unit_input`` preserves the original unit provided by the user.
- Index return series are always treated as monthly returns; the config return
  unit does not apply to the index series.
- Breach thresholds compare monthly returns; shortfall thresholds compare an
  annualised compounded return hurdle.
- Summary tables report annualised terminal_AnnReturn/monthly_AnnVol/monthly_TE;
  probabilities and counts are unitless.
"""

from __future__ import annotations

from typing import Literal, Mapping, cast

import numpy as np
import pandas as pd

from .config import (
    CANONICAL_RETURN_UNIT,
    DEFAULT_MEAN_CONVERSION,
    MONTHS_PER_YEAR,
    ModelConfig,
    annual_cov_to_monthly,
    annual_mean_to_monthly,
    annual_vol_to_monthly,
)

Unit = Literal["annual", "monthly"]

DEFAULT_BREACH_THRESHOLD = -0.02
DEFAULT_SHORTFALL_THRESHOLD = -0.05
SUMMARY_TABLE_UNIT: Unit = "annual"

__all__ = [
    "CONFIG_TIME_HORIZON_FIELDS",
    "DEFAULT_BREACH_THRESHOLD",
    "DEFAULT_SHORTFALL_THRESHOLD",
    "SUMMARY_TABLE_UNIT",
    "Unit",
    "convert_annual_series_to_monthly",
    "convert_covariance",
    "convert_mean",
    "convert_return_series",
    "convert_volatility",
    "format_unit_label",
    "get_config_unit",
    "get_index_series_unit",
    "normalize_return_inputs",
    "get_summary_table_unit",
    "get_threshold_unit",
    "normalize_index_series",
    "annual_mean_to_monthly",
    "annual_vol_to_monthly",
    "annual_cov_to_monthly",
]

# Audit list of config fields whose meaning depends on time horizon.
# Units are expressed as: annual, monthly, months, or return_unit_input.
CONFIG_TIME_HORIZON_FIELDS: dict[str, str] = {
    "N_MONTHS": "months",
    "return_unit": "annual_or_monthly",
    "return_unit_input": "annual_or_monthly",
    "mu_H": "return_unit_input",
    "mu_H_annual": "annual",
    "mu_H_monthly": "monthly",
    "sigma_H": "return_unit_input",
    "sigma_H_annual": "annual",
    "sigma_H_monthly": "monthly",
    "mu_E": "return_unit_input",
    "mu_E_annual": "annual",
    "mu_E_monthly": "monthly",
    "sigma_E": "return_unit_input",
    "sigma_E_annual": "annual",
    "sigma_E_monthly": "monthly",
    "mu_M": "return_unit_input",
    "mu_M_annual": "annual",
    "mu_M_monthly": "monthly",
    "sigma_M": "return_unit_input",
    "sigma_M_annual": "annual",
    "sigma_M_monthly": "monthly",
    "in_house_return_min_pct": "annual",
    "in_house_return_max_pct": "annual",
    "in_house_return_step_pct": "annual",
    "in_house_vol_min_pct": "annual",
    "in_house_vol_max_pct": "annual",
    "in_house_vol_step_pct": "annual",
    "alpha_ext_return_min_pct": "annual",
    "alpha_ext_return_max_pct": "annual",
    "alpha_ext_return_step_pct": "annual",
    "alpha_ext_vol_min_pct": "annual",
    "alpha_ext_vol_max_pct": "annual",
    "alpha_ext_vol_step_pct": "annual",
    "internal_financing_mean_month": "monthly",
    "internal_financing_sigma_month": "monthly",
    "internal_spike_prob": "monthly",
    "ext_pa_financing_mean_month": "monthly",
    "ext_pa_financing_sigma_month": "monthly",
    "ext_pa_spike_prob": "monthly",
    "act_ext_financing_mean_month": "monthly",
    "act_ext_financing_sigma_month": "monthly",
    "act_ext_spike_prob": "monthly",
    "vol_regime_window": "months",
    "reference_sigma": "monthly",
    "financing_term_months": "months",
}


def convert_annual_series_to_monthly(
    series: pd.Series,
    *,
    method: Literal["simple", "geometric"] = DEFAULT_MEAN_CONVERSION,
) -> pd.Series:
    """Convert an annualized return series to monthly returns.

    ``series`` is expressed in annual units; the returned series is monthly.
    """
    if method == "geometric":
        return (1.0 + series) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    return series / MONTHS_PER_YEAR


def _coerce_unit(unit: str) -> Unit:
    if unit not in ("annual", "monthly"):
        raise ValueError(f"unit must be 'annual' or 'monthly', got {unit!r}")
    return cast(Unit, unit)


def format_unit_label(unit: Unit) -> str:
    """Return a human-friendly label for ``unit`` suitable for text output."""
    return "annualised" if unit == "annual" else "monthly"


def convert_mean(
    value: float,
    *,
    from_unit: Unit,
    to_unit: Unit,
    method: Literal["simple", "geometric"] = DEFAULT_MEAN_CONVERSION,
) -> float:
    """Convert a mean return from ``from_unit`` to ``to_unit``."""
    from_unit = _coerce_unit(from_unit)
    to_unit = _coerce_unit(to_unit)
    if from_unit == to_unit:
        return float(value)
    if from_unit == "annual":
        return annual_mean_to_monthly(value, method=method)
    if method == "geometric":
        return float((1.0 + value) ** MONTHS_PER_YEAR - 1.0)
    return float(value * MONTHS_PER_YEAR)


def convert_volatility(value: float, *, from_unit: Unit, to_unit: Unit) -> float:
    """Convert a volatility from ``from_unit`` to ``to_unit``."""
    from_unit = _coerce_unit(from_unit)
    to_unit = _coerce_unit(to_unit)
    if from_unit == to_unit:
        return float(value)
    if from_unit == "annual":
        return float(annual_vol_to_monthly(value))
    return float(value * (MONTHS_PER_YEAR**0.5))


def convert_covariance(cov: object, *, from_unit: Unit, to_unit: Unit) -> np.ndarray:
    """Convert a covariance matrix from ``from_unit`` to ``to_unit``."""
    from_unit = _coerce_unit(from_unit)
    to_unit = _coerce_unit(to_unit)
    if from_unit == to_unit:
        return np.asarray(cov, dtype=float)
    if from_unit == "annual":
        return annual_cov_to_monthly(cov)  # type: ignore[arg-type]
    return np.asarray(cov, dtype=float) * MONTHS_PER_YEAR


def convert_return_series(
    series: pd.Series,
    *,
    from_unit: Unit,
    to_unit: Unit,
    method: Literal["simple", "geometric"] = DEFAULT_MEAN_CONVERSION,
) -> pd.Series:
    """Convert a return series from ``from_unit`` to ``to_unit``."""
    from_unit = _coerce_unit(from_unit)
    to_unit = _coerce_unit(to_unit)
    if from_unit == to_unit:
        return pd.Series(series)
    if from_unit == "annual":
        return convert_annual_series_to_monthly(series, method=method)
    if method == "geometric":
        return (1.0 + series) ** MONTHS_PER_YEAR - 1.0
    return series * MONTHS_PER_YEAR


def normalize_index_series(idx_series: pd.Series, input_unit: str) -> pd.Series:
    """Return a monthly index series given an ``input_unit`` label."""
    unit = input_unit or CANONICAL_RETURN_UNIT
    series = pd.Series(idx_series)
    if unit == "annual":
        return convert_annual_series_to_monthly(series)
    return series


def get_config_unit(cfg: object | None = None) -> Unit:
    """Return the unit for config return parameters after normalization (monthly)."""
    if cfg is None:
        return CANONICAL_RETURN_UNIT
    return _coerce_unit(getattr(cfg, "return_unit", CANONICAL_RETURN_UNIT))


def get_index_series_unit() -> Unit:
    """Return the expected unit for index return series (monthly)."""
    return CANONICAL_RETURN_UNIT


def normalize_return_inputs(cfg: ModelConfig) -> dict[str, float]:
    """Return monthly return inputs derived from ``cfg``."""
    unit = get_config_unit(cfg)
    return {
        "mu_H": convert_mean(cfg.mu_H, from_unit=unit, to_unit="monthly"),
        "mu_E": convert_mean(cfg.mu_E, from_unit=unit, to_unit="monthly"),
        "mu_M": convert_mean(cfg.mu_M, from_unit=unit, to_unit="monthly"),
        "sigma_H": convert_volatility(cfg.sigma_H, from_unit=unit, to_unit="monthly"),
        "sigma_E": convert_volatility(cfg.sigma_E, from_unit=unit, to_unit="monthly"),
        "sigma_M": convert_volatility(cfg.sigma_M, from_unit=unit, to_unit="monthly"),
    }


def get_threshold_unit() -> Mapping[str, Unit]:
    """Return units for thresholds (breach monthly, shortfall annual compounded)."""
    return {
        "breach_threshold": "monthly",
        "shortfall_threshold": "annual",
    }


def get_summary_table_unit(periods_per_year: int | None = None) -> Unit:
    """Return the unit for summary table terminal_AnnReturn/monthly_AnnVol/monthly_TE outputs.

    Summary tables are annualised regardless of ``return_unit`` inputs. When
    ``periods_per_year`` is 1, the outputs are effectively monthly.
    """
    if periods_per_year == 1:
        return "monthly"
    return SUMMARY_TABLE_UNIT
