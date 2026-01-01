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

    def fake_draw_joint_returns(*, n_months, n_sim, params, rng=None, shocks=None):
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

    def capture_cli_draw_joint_returns(*, n_months, n_sim, params, rng=None, shocks=None):
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
