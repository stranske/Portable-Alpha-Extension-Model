from pa_core.config import ModelConfig, load_config
from pa_core.sweep import generate_parameter_combinations


def _config_with_sweep(sweep_data: dict) -> ModelConfig:
    base = load_config("examples/scenarios/my_first_scenario.yml")
    data = base.model_dump()
    data["sweep"] = sweep_data
    return ModelConfig.model_validate(data)


def test_generate_parameter_combinations_grid_sweep() -> None:
    cfg = _config_with_sweep(
        {
            "method": "grid",
            "parameters": {
                "mu_H": {"values": [0.01, 0.02]},
                "sigma_H": {"min": 0.01, "max": 0.03, "step": 0.01},
            },
        }
    )

    combos = list(generate_parameter_combinations(cfg))

    assert len(combos) == 6
    for combo in combos:
        assert set(combo.keys()) == {"mu_H", "sigma_H"}
        assert combo["mu_H"] in {0.01, 0.02}
        assert combo["sigma_H"] in {0.01, 0.02, 0.03}


def test_generate_parameter_combinations_random_sweep_deterministic() -> None:
    cfg = _config_with_sweep(
        {
            "method": "random",
            "samples": 3,
            "seed": 11,
            "parameters": {
                "theta_extpa": {"values": [0.2, 0.4, 0.6]},
                "active_share": {"min": 0.1, "max": 0.2},
            },
        }
    )

    combos_first = list(generate_parameter_combinations(cfg))
    combos_second = list(generate_parameter_combinations(cfg))

    assert combos_first == combos_second
    assert len(combos_first) == 3
    for combo in combos_first:
        assert combo["theta_extpa"] in {0.2, 0.4, 0.6}
        assert 0.1 <= combo["active_share"] <= 0.2
