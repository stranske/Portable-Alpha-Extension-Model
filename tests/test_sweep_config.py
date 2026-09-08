import pandas as pd
import pytest

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


@pytest.mark.parametrize("cached", [False, True])
def test_sweep_theta_endpoints_match_validated_controls(cached):
    from pa_core.random import spawn_agent_rngs, spawn_rngs
    from pa_core.sweep import clear_sweep_cache, run_parameter_sweep, run_parameter_sweep_cached

    cfg = ModelConfig(
        N_SIMULATIONS=100,
        N_MONTHS=6,
        external_pa_capital=200,
        mu_M=0.36,
        sigma_M=0.03,
        theta_extpa=0.5,
        financing_mode="broadcast",
        sweep={"method": "grid", "parameters": {"theta_extpa": {"values": [0.0, 1.0]}}},
    )
    idx = pd.Series([0.01, -0.02, 0.03, 0.0, 0.02, -0.01])

    def run(config):
        if cached:
            return run_parameter_sweep_cached(config, idx, seed=73)
        return run_parameter_sweep(
            config,
            idx,
            spawn_rngs(73, 1)[0],
            spawn_agent_rngs(73, ["internal", "external_pa", "active_ext"]),
            seed=73,
        )

    clear_sweep_cache()
    results = run(cfg)
    if cached:
        assert run(cfg) is results
    for result in results:
        assert result["summary"]["Agent"].tolist()[0] == "Base"
        data = cfg.model_dump()
        data.update(result["parameters"])
        data["sweep"] = {
            "method": "grid",
            "parameters": {"theta_extpa": {"values": [data["theta_extpa"]]}},
        }
        control = ModelConfig.model_validate(data)
        pd.testing.assert_frame_equal(result["summary"], run(control)[0]["summary"])
    endpoint_returns = [
        row["summary"].set_index("Agent").loc["ExternalPA", "terminal_AnnReturn"] for row in results
    ]
    assert abs(endpoint_returns[1] - endpoint_returns[0]) > 0.01
    clear_sweep_cache()


def test_agent_overrides_preserve_explicit_agents_and_monthly_inputs():
    cfg = ModelConfig(
        N_SIMULATIONS=100,
        N_MONTHS=6,
        financing_mode="broadcast",
        mu_M_annual=0.36,
        sigma_M_annual=0.12,
        agents=[
            {"name": "Base", "capital": 1000, "beta_share": 0.6, "alpha_share": 0.4},
            {
                "name": "CustomSleeve",
                "capital": 50,
                "beta_share": 0.05,
                "alpha_share": 0,
                "extra": {"custom_setting": 7},
            },
        ],
    )
    updated = cfg.with_agent_overrides({"external_pa_capital": 200, "theta_extpa": 0.8})
    assert [agent.name for agent in updated.agents] == ["Base", "CustomSleeve", "ExternalPA"]
    agents = {agent.name: agent for agent in updated.agents}
    assert agents["Base"] == cfg.agents[0]
    assert agents["CustomSleeve"] == cfg.agents[1]
    assert agents["ExternalPA"].capital == 200
    assert agents["ExternalPA"].beta_share == pytest.approx(0.2)
    assert agents["ExternalPA"].extra["theta_extpa"] == 0.8
    assert updated.mu_M == cfg.mu_M == pytest.approx(0.03)
    assert updated.sigma_M == cfg.sigma_M == pytest.approx(0.12 / 12**0.5)
    assert updated.return_unit_input == cfg.return_unit_input == "annual"
    assert [agent.name for agent in cfg.agents] == ["Base", "CustomSleeve"]
    unchanged = cfg.with_agent_overrides({"mu_M": 0.02})
    assert unchanged.agents == cfg.agents


def test_agent_overrides_keep_over_margin_candidates_and_drop_zero_sleeves():
    cfg = ModelConfig(
        N_SIMULATIONS=100, N_MONTHS=6, external_pa_capital=100, financing_mode="broadcast"
    )
    updated = cfg.with_agent_overrides({"external_pa_capital": 0, "internal_pa_capital": 1000})
    assert next(a for a in updated.agents if a.name == "InternalPA").capital == 1000
    with pytest.raises(ValueError, match="[Mm]argin"):
        ModelConfig.model_validate(updated.model_dump())
    zero = updated.with_agent_overrides({"internal_pa_capital": 0})
    assert [a.name for a in zero.agents] == ["Base"]


@pytest.mark.parametrize(
    "updates",
    [
        {"theta_extpa": 0.8},
        {"active_share": 0.7},
        {"w_beta_H": 0.6, "w_alpha_H": 0.4},
        {"total_fund_capital": 1200},
    ],
)
def test_agent_overrides_replace_affected_agents_in_place(updates):
    cfg = ModelConfig(
        N_SIMULATIONS=100,
        N_MONTHS=6,
        external_pa_capital=100,
        active_ext_capital=100,
        internal_pa_capital=100,
        financing_mode="broadcast",
    )
    custom = cfg.agents[0].model_copy(update={"name": "CustomSleeve"})
    # Custom input order is intentional and must survive any subset refresh.
    cfg = cfg.model_copy(
        update={"agents": [cfg.agents[2], custom, cfg.agents[0], cfg.agents[1], cfg.agents[3]]}
    )
    updated = cfg.with_agent_overrides(updates)
    assert [agent.name for agent in updated.agents] == [agent.name for agent in cfg.agents]
    assert updated.agents[1] is custom
