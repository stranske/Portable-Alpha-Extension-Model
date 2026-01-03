from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pa_core.cli import Dependencies, main
from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.random import spawn_agent_rngs, spawn_rngs
from pa_core.sweep import run_parameter_sweep


def test_entrypoints_keep_index_series_monthly(tmp_path: Path, monkeypatch) -> None:
    cfg_data = {
        "N_SIMULATIONS": 2,
        "N_MONTHS": 3,
        "financing_mode": "broadcast",
        "return_unit": "annual",
        "analysis_mode": "returns",
        "in_house_return_min_pct": 2.0,
        "in_house_return_max_pct": 2.0,
        "in_house_return_step_pct": 1.0,
        "in_house_vol_min_pct": 1.0,
        "in_house_vol_max_pct": 1.0,
        "in_house_vol_step_pct": 1.0,
        "alpha_ext_return_min_pct": 1.0,
        "alpha_ext_return_max_pct": 1.0,
        "alpha_ext_return_step_pct": 1.0,
        "alpha_ext_vol_min_pct": 2.0,
        "alpha_ext_vol_max_pct": 2.0,
        "alpha_ext_vol_step_pct": 1.0,
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))

    idx_values = [0.12, 0.0, -0.06, 0.12]
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n" + "\n".join(str(val) for val in idx_values))
    idx_series = pd.Series(idx_values)
    expected_mu = float(idx_series.mean())

    def fake_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        params_capture["params"] = dict(params)
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros

    def fake_simulate_agents(*_args, **_kwargs):
        return {"Base": np.zeros((2, 3))}

    params_capture: dict[str, dict[str, object]] = {}
    monkeypatch.setattr("pa_core.orchestrator.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.orchestrator.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.orchestrator.simulate_agents", fake_simulate_agents)

    cfg = load_config(cfg_path)
    orch = SimulatorOrchestrator(cfg, idx_series)
    orch.run(seed=7)
    assert params_capture["params"]["mu_idx_month"] == expected_mu

    params_capture.clear()
    monkeypatch.setattr("pa_core.sweep.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.sweep.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sweep.simulate_agents", fake_simulate_agents)

    rng_returns = spawn_rngs(7, 1)[0]
    fin_rngs = spawn_agent_rngs(7, ["internal", "external_pa", "active_ext"])
    run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)
    assert params_capture["params"]["mu_idx_month"] == expected_mu

    cli_params: dict[str, object] = {}

    def capture_cli_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        cli_params.update(dict(params))
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def noop_export(*_args, **_kwargs) -> None:
        return None

    deps = Dependencies(
        draw_joint_returns=capture_cli_draw_joint_returns,
        draw_financing_series=fake_draw_financing_series,
        simulate_agents=fake_simulate_agents,
        export_to_excel=noop_export,
    )

    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_path),
            "--output",
            str(tmp_path / "out.xlsx"),
            "--seed",
            "7",
            "--sensitivity",
        ],
        deps=deps,
    )
    assert cli_params["mu_idx_month"] == expected_mu


def test_entrypoints_summary_table_annualised(tmp_path: Path, monkeypatch) -> None:
    cfg_data_single = {
        "N_SIMULATIONS": 2,
        "N_MONTHS": 12,
        "financing_mode": "broadcast",
        "return_unit": "annual",
        "analysis_mode": "single_with_sensitivity",
    }
    cfg_single_path = tmp_path / "cfg_single.yml"
    cfg_single_path.write_text(yaml.safe_dump(cfg_data_single))

    cfg_data_sweep = {
        "N_SIMULATIONS": 2,
        "N_MONTHS": 12,
        "financing_mode": "broadcast",
        "return_unit": "annual",
        "analysis_mode": "returns",
        "in_house_return_min_pct": 1.0,
        "in_house_return_max_pct": 1.0,
        "in_house_return_step_pct": 1.0,
        "in_house_vol_min_pct": 1.0,
        "in_house_vol_max_pct": 1.0,
        "in_house_vol_step_pct": 1.0,
        "alpha_ext_return_min_pct": 1.0,
        "alpha_ext_return_max_pct": 1.0,
        "alpha_ext_return_step_pct": 1.0,
        "alpha_ext_vol_min_pct": 1.0,
        "alpha_ext_vol_max_pct": 1.0,
        "alpha_ext_vol_step_pct": 1.0,
    }
    cfg_sweep_path = tmp_path / "cfg_sweep.yml"
    cfg_sweep_path.write_text(yaml.safe_dump(cfg_data_sweep))

    idx_values = [0.01] * 12
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n" + "\n".join(str(val) for val in idx_values))
    idx_series = pd.Series(idx_values)

    monthly_return = 0.01
    expected_ann = (1.0 + monthly_return) ** 12 - 1.0
    returns_map = {"Base": np.full((2, 12), monthly_return)}

    def fake_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros

    def fake_simulate_agents(*_args, **_kwargs):
        return returns_map

    cfg = load_config(cfg_single_path)
    cfg_sweep = load_config(cfg_sweep_path)

    monkeypatch.setattr("pa_core.orchestrator.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.orchestrator.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.orchestrator.simulate_agents", fake_simulate_agents)

    _, summary = SimulatorOrchestrator(cfg, idx_series).run(seed=1)
    base_row = summary[summary["Agent"] == "Base"].iloc[0]
    assert np.isclose(base_row["terminal_AnnReturn"], expected_ann)

    monkeypatch.setattr("pa_core.sweep.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.sweep.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sweep.simulate_agents", fake_simulate_agents)

    rng_returns = spawn_rngs(1, 1)[0]
    fin_rngs = spawn_agent_rngs(1, ["internal", "external_pa", "active_ext"])
    results = run_parameter_sweep(cfg_sweep, idx_series, rng_returns, fin_rngs)
    sweep_row = results[0]["summary"][results[0]["summary"]["Agent"] == "Base"].iloc[0]
    assert np.isclose(sweep_row["terminal_AnnReturn"], expected_ann)

    captured: dict[str, object] = {}

    def capture_summary(summary_df):
        captured["summary"] = summary_df

    def noop_export(*_args, **_kwargs) -> None:
        return None

    def fake_build_from_config(_cfg):
        return []

    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", capture_summary)

    deps = Dependencies(
        draw_joint_returns=fake_draw_joint_returns,
        draw_financing_series=fake_draw_financing_series,
        simulate_agents=fake_simulate_agents,
        export_to_excel=noop_export,
        build_from_config=fake_build_from_config,
    )

    main(
        [
            "--config",
            str(cfg_single_path),
            "--index",
            str(idx_path),
            "--output",
            str(tmp_path / "out.xlsx"),
            "--seed",
            "1",
            "--sensitivity",
        ],
        deps=deps,
    )
    cli_summary = captured["summary"]
    base_row = cli_summary[cli_summary["Agent"] == "Base"].iloc[0]
    assert np.isclose(base_row["terminal_AnnReturn"], expected_ann)


def test_sweep_return_overrides_convert_to_monthly(monkeypatch) -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 1,
            "N_MONTHS": 2,
            "financing_mode": "broadcast",
            "return_unit": "annual",
            "analysis_mode": "returns",
            "in_house_return_min_pct": 2.0,
            "in_house_return_max_pct": 2.0,
            "in_house_return_step_pct": 1.0,
            "in_house_vol_min_pct": 1.0,
            "in_house_vol_max_pct": 1.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_min_pct": 1.0,
            "alpha_ext_return_max_pct": 1.0,
            "alpha_ext_return_step_pct": 1.0,
            "alpha_ext_vol_min_pct": 2.0,
            "alpha_ext_vol_max_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
        }
    )
    idx_series = pd.Series([0.0, 0.0])

    params_capture: dict[str, dict[str, object]] = {}

    def fake_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        params_capture["params"] = dict(params)
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros

    def fake_simulate_agents(*_args, **_kwargs):
        return {"Base": np.zeros((1, 2))}

    monkeypatch.setattr("pa_core.sweep.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.sweep.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sweep.simulate_agents", fake_simulate_agents)

    rng_returns = spawn_rngs(7, 1)[0]
    fin_rngs = spawn_agent_rngs(7, ["internal", "external_pa", "active_ext"])
    run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)

    params = params_capture["params"]
    assert np.isclose(params["default_mu_H"], 0.02 / 12.0)
    assert np.isclose(params["default_sigma_H"], 0.01 / np.sqrt(12.0))
    assert np.isclose(params["default_mu_E"], 0.01 / 12.0)
    assert np.isclose(params["default_sigma_E"], 0.02 / np.sqrt(12.0))
