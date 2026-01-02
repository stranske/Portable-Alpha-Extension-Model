import numpy as np
import pandas as pd

from pa_core.config import (
    ModelConfig,
    annual_cov_to_monthly,
    annual_mean_to_monthly,
    annual_vol_to_monthly,
)
from pa_core.sim.metrics import annualised_return
from pa_core.sim.params import build_simulation_params
from pa_core.sim.paths import draw_joint_returns
from pa_core.units import (
    convert_annual_series_to_monthly,
    convert_mean,
    convert_return_series,
    convert_volatility,
    format_unit_label,
    get_config_unit,
    get_index_series_unit,
    get_summary_table_unit,
    get_threshold_unit,
    normalize_index_series,
)


def test_annual_mean_to_monthly_simple_and_geometric() -> None:
    annual = 0.12
    assert annual_mean_to_monthly(annual) == annual / 12.0
    geometric = annual_mean_to_monthly(annual, method="geometric")
    expected = (1.0 + annual) ** (1.0 / 12.0) - 1.0
    assert np.isclose(geometric, expected)


def test_convert_annual_series_to_monthly_simple_and_geometric() -> None:
    series = pd.Series([0.12, -0.06])
    expected_simple = series / 12.0
    assert np.allclose(convert_annual_series_to_monthly(series), expected_simple)

    expected_geom = (1.0 + series) ** (1.0 / 12.0) - 1.0
    converted_geom = convert_annual_series_to_monthly(series, method="geometric")
    assert np.allclose(converted_geom, expected_geom)


def test_convert_mean_round_trip_simple() -> None:
    monthly = 0.01
    annual = convert_mean(monthly, from_unit="monthly", to_unit="annual", method="simple")
    assert np.isclose(annual, 0.12)
    back = convert_mean(annual, from_unit="annual", to_unit="monthly", method="simple")
    assert np.isclose(back, monthly)


def test_convert_mean_annual_point05_to_monthly() -> None:
    annual = 0.05
    expected = annual / 12.0
    monthly = convert_mean(annual, from_unit="annual", to_unit="monthly", method="simple")
    assert np.isclose(monthly, expected)


def test_convert_return_series_monthly_to_annual() -> None:
    series = pd.Series([0.01, -0.02])
    converted = convert_return_series(series, from_unit="monthly", to_unit="annual")
    expected = series * 12.0
    assert np.allclose(converted, expected)


def test_convert_volatility_round_trip() -> None:
    monthly = 0.02
    annual = convert_volatility(monthly, from_unit="monthly", to_unit="annual")
    expected = monthly * np.sqrt(12.0)
    assert np.isclose(annual, expected)
    back = convert_volatility(annual, from_unit="annual", to_unit="monthly")
    assert np.isclose(back, monthly)


def test_normalize_index_series_converts_when_annual() -> None:
    series = pd.Series([0.12, 0.0, -0.06])
    expected = convert_annual_series_to_monthly(series)
    converted = normalize_index_series(series, "annual")
    assert np.allclose(converted, expected)


def test_normalize_index_series_passthrough_monthly() -> None:
    series = pd.Series([0.01, 0.02])
    converted = normalize_index_series(series, "monthly")
    assert np.allclose(converted, series)


def test_normalize_index_series_defaults_to_monthly() -> None:
    series = pd.Series([0.01, -0.03])
    converted = normalize_index_series(series, "")
    assert np.allclose(converted, series)


def test_unit_policy_accessors() -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)
    assert get_config_unit(cfg) == "monthly"
    assert get_index_series_unit() == "monthly"
    assert get_summary_table_unit() == "annual"
    assert get_summary_table_unit(periods_per_year=1) == "monthly"
    units = get_threshold_unit()
    assert units["breach_threshold"] == "monthly"
    assert units["shortfall_threshold"] == "annual"
    assert format_unit_label("annual") == "annualised"


def test_annual_vol_to_monthly() -> None:
    annual = 0.24
    expected = annual / np.sqrt(12.0)
    assert np.isclose(annual_vol_to_monthly(annual), expected)


def test_annual_cov_to_monthly() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    expected = cov / 12.0
    converted = annual_cov_to_monthly(cov)
    assert np.allclose(converted, expected)


def test_annual_cov_matches_vol_conversion() -> None:
    annual_vols = np.array([0.24, 0.12])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    annual_cov = np.outer(annual_vols, annual_vols) * corr
    monthly_cov = annual_cov_to_monthly(annual_cov)
    monthly_vols = np.array([annual_vol_to_monthly(v) for v in annual_vols])
    expected = np.outer(monthly_vols, monthly_vols) * corr
    assert np.allclose(monthly_cov, expected)


def test_model_config_converts_once() -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, mu_H=0.12, sigma_H=0.24)
    assert np.isclose(cfg.mu_H, annual_mean_to_monthly(0.12))
    assert np.isclose(cfg.sigma_H, annual_vol_to_monthly(0.24))
    assert cfg.return_unit == "monthly"

    round_trip = cfg.__class__.model_validate(cfg.model_dump())
    assert np.isclose(round_trip.mu_H, cfg.mu_H)
    assert np.isclose(round_trip.sigma_H, cfg.sigma_H)


def test_model_config_monthly_passthrough() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        return_unit="monthly",
        mu_H=0.01,
        sigma_H=0.02,
    )
    assert cfg.return_unit == "monthly"
    assert cfg.return_unit_input == "monthly"
    assert np.isclose(cfg.mu_H, 0.01)
    assert np.isclose(cfg.sigma_H, 0.02)


def test_annual_config_round_trip_to_annual_output() -> None:
    annual_mu = 0.12
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=12,
        mu_H=annual_mu,
        sigma_H=0.0,
        mu_E=0.0,
        sigma_E=0.0,
        mu_M=0.0,
        sigma_M=0.0,
    )
    params = build_simulation_params(cfg, mu_idx=0.0, idx_sigma=0.0)
    r_beta, r_H, _r_E, _r_M = draw_joint_returns(
        n_months=cfg.N_MONTHS,
        n_sim=cfg.N_SIMULATIONS,
        params=params,
        rng=np.random.default_rng(123),
    )
    assert np.allclose(r_beta, 0.0)
    assert np.allclose(r_H, cfg.mu_H)
    annual_out = annualised_return(r_H, periods_per_year=12)
    expected = (1.0 + (annual_mu / 12.0)) ** 12 - 1.0
    assert np.isclose(annual_out, expected)
