import pytest

from pa_core.config import ModelConfig, annual_mean_to_monthly
from pa_core.stress import apply_stress_preset


def _base_cfg() -> ModelConfig:
    return ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")


def test_liquidity_squeeze_overrides_financing():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "liquidity_squeeze")
    assert stressed.internal_spike_prob == pytest.approx(0.1)
    assert stressed.ext_pa_spike_factor == pytest.approx(5.0)


def test_2008_vol_regime_triples_vol():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "2008_vol_regime")
    assert stressed.sigma_H == pytest.approx(cfg.sigma_H * 3)
    assert stressed.sigma_E == pytest.approx(cfg.sigma_E * 3)


def test_invalid_preset_raises():
    cfg = _base_cfg()
    with pytest.raises(KeyError):
        apply_stress_preset(cfg, "unknown")


def test_label_alias_resolves():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "Liquidity squeeze")
    assert stressed.internal_spike_prob == pytest.approx(0.1)


def test_correlation_breakdown_overrides_rhos():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "correlation_breakdown")
    assert stressed.rho_idx_H == pytest.approx(0.95)
    assert stressed.rho_H_E == pytest.approx(0.95)
    assert stressed.rho_E_M == pytest.approx(0.95)


def test_2020_gap_day_overrides_returns():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "2020_gap_day")
    assert stressed.mu_H == pytest.approx(annual_mean_to_monthly(-0.20))
    assert stressed.mu_E == pytest.approx(annual_mean_to_monthly(-0.25))
    assert stressed.mu_M == pytest.approx(annual_mean_to_monthly(-0.15))


def test_rate_shock_overrides_financing_means():
    cfg = _base_cfg()
    stressed = apply_stress_preset(cfg, "rate_shock")
    assert stressed.internal_financing_mean_month == pytest.approx(0.012)
    assert stressed.ext_pa_financing_mean_month == pytest.approx(0.014)
    assert stressed.act_ext_financing_mean_month == pytest.approx(0.015)
