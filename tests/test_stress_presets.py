import pytest

from pa_core.config import ModelConfig
from pa_core.stress import apply_stress_preset


def _base_cfg() -> ModelConfig:
    return ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)


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
