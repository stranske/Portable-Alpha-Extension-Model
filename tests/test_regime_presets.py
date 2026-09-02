import pytest

from pa_core.config import ModelConfig
from pa_core.regime_presets import REGIME_PRESET_LABELS, apply_regime_preset


def _config() -> ModelConfig:
    return ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        return_unit="monthly",
        sigma_H=0.10,
        sigma_E=0.20,
        sigma_M=0.30,
    )


@pytest.mark.parametrize(
    (
        "name",
        "expected_key",
        "expected_label",
        "expected_regime",
        "expected_multiplier",
        "expected_sigma_m_multiplier",
        "expected_transition",
    ),
    [
        (
            "2008_crisis",
            "2008_crisis",
            "2008 crisis",
            "crisis",
            2.5,
            2.5,
            [[0.95, 0.05], [0.2, 0.8]],
        ),
        (
            "2008 crisis",
            "2008_crisis",
            "2008 crisis",
            "crisis",
            2.5,
            2.5,
            [[0.95, 0.05], [0.2, 0.8]],
        ),
        (
            "COVID-SHOCK",
            "covid_shock",
            "COVID shock",
            "shock",
            3.0,
            2.5,
            [[0.97, 0.03], [0.6, 0.4]],
        ),
    ],
)
def test_apply_regime_preset_builds_named_stress_regime(
    name: str,
    expected_key: str,
    expected_label: str,
    expected_regime: str,
    expected_multiplier: float,
    expected_sigma_m_multiplier: float,
    expected_transition: list[list[float]],
) -> None:
    source = _config()
    source_snapshot = source.model_dump()
    expected_sigma_h = source.sigma_H * expected_multiplier
    expected_sigma_e = source.sigma_E * expected_multiplier
    expected_sigma_m = source.sigma_M * expected_sigma_m_multiplier

    configured = apply_regime_preset(source, name)
    label = REGIME_PRESET_LABELS[expected_key]
    configured_from_label = apply_regime_preset(source, label)

    assert label == expected_label
    assert configured_from_label.regime_transition == configured.regime_transition
    assert configured is not source
    assert source.model_dump() == source_snapshot
    assert configured.regime_start == "calm"
    assert configured.regime_transition == expected_transition
    assert configured.regimes is not None
    assert [regime.name for regime in configured.regimes] == ["calm", expected_regime]
    stress = configured.regimes[1]
    assert stress.idx_sigma_multiplier == expected_multiplier
    assert stress.sigma_H == pytest.approx(expected_sigma_h)
    assert stress.sigma_E == pytest.approx(expected_sigma_e)
    assert stress.sigma_M == pytest.approx(expected_sigma_m)


def test_apply_regime_preset_rejects_unknown_or_blank_names() -> None:
    for name in ("", "  ", "financial crisis"):
        with pytest.raises(KeyError, match=f"Unknown regime preset: {name}"):
            apply_regime_preset(_config(), name)
