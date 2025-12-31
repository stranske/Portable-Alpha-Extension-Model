from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pa_core.cli import Dependencies, main
from pa_core.config import load_config
from pa_core.data import load_index_returns
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core import simulations as sim_module
from pa_core.sim.paths import draw_joint_returns as draw_joint_returns_impl


def test_cli_and_orchestrator_draws_match(tmp_path: Path, monkeypatch) -> None:
    cfg_data = {
        "N_SIMULATIONS": 4,
        "N_MONTHS": 3,
        "analysis_mode": "single_with_sensitivity",
        "w_beta_H": 0.6,
        "w_alpha_H": 0.4,
        "risk_metrics": ["ShortfallProb"],
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))

    idx_values = [0.01, 0.02, 0.015, 0.03, 0.005, 0.025]
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n" + "\n".join(str(val) for val in idx_values))
    idx_series = load_index_returns(idx_path)

    seed = 123
    orch_capture: dict[str, tuple[np.ndarray, ...]] = {}

    def capture_orch_simulate_agents(
        agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act
    ):
        orch_capture["draws"] = (
            np.array(r_beta),
            np.array(r_H),
            np.array(r_E),
            np.array(r_M),
        )
        return sim_module.simulate_agents(
            agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act
        )

    orch_params: dict[str, object] = {}

    def capture_orch_draw_joint_returns(*, n_months, n_sim, params, rng=None, shocks=None):
        if "params" not in orch_params:
            orch_params["params"] = dict(params)
            if rng is not None:
                orch_params["rng_state"] = rng.bit_generator.state
        return draw_joint_returns_impl(
            n_months=n_months, n_sim=n_sim, params=params, rng=rng, shocks=shocks
        )

    monkeypatch.setattr(
        "pa_core.orchestrator.simulate_agents", capture_orch_simulate_agents
    )
    monkeypatch.setattr(
        "pa_core.orchestrator.draw_joint_returns", capture_orch_draw_joint_returns
    )

    cfg = load_config(cfg_path)
    orch = SimulatorOrchestrator(cfg, idx_series)
    orch.run(seed=seed)

    cli_capture: dict[str, tuple[np.ndarray, ...]] = {}

    def capture_cli_simulate_agents(
        agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act
    ):
        if "draws" not in cli_capture:
            cli_capture["draws"] = (
                np.array(r_beta),
                np.array(r_H),
                np.array(r_E),
                np.array(r_M),
            )
        return sim_module.simulate_agents(
            agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act
        )

    cli_params: dict[str, object] = {}

    def capture_cli_draw_joint_returns(*, n_months, n_sim, params, rng=None, shocks=None):
        if "params" not in cli_params:
            cli_params["params"] = dict(params)
            if rng is not None:
                cli_params["rng_state"] = rng.bit_generator.state
        return draw_joint_returns_impl(
            n_months=n_months, n_sim=n_sim, params=params, rng=rng, shocks=shocks
        )

    def noop_export(*_args, **_kwargs) -> None:
        return None

    deps = Dependencies(
        simulate_agents=capture_cli_simulate_agents,
        export_to_excel=noop_export,
        draw_joint_returns=capture_cli_draw_joint_returns,
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
            str(seed),
            "--sensitivity",
        ],
        deps=deps,
    )

    assert "draws" in orch_capture
    assert "draws" in cli_capture
    assert "params" in orch_params
    assert "params" in cli_params
    assert orch_params["params"].keys() <= cli_params["params"].keys()
    for key in orch_params["params"]:
        orch_val = orch_params["params"][key]
        cli_val = cli_params["params"][key]
        if isinstance(orch_val, str) or isinstance(cli_val, str):
            assert orch_val == cli_val
        else:
            np.testing.assert_allclose(
                np.array(orch_val, dtype=float),
                np.array(cli_val, dtype=float),
            )
    assert orch_params.get("rng_state") == cli_params.get("rng_state")
    for orch_arr, cli_arr in zip(orch_capture["draws"], cli_capture["draws"]):
        np.testing.assert_allclose(orch_arr, cli_arr)
