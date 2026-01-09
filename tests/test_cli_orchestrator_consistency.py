from pathlib import Path

import numpy as np
import yaml

from pa_core import simulations as sim_module
from pa_core.cli import main
from pa_core.config import load_config
from pa_core.data import load_index_returns
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.sim.paths import draw_joint_returns as draw_joint_returns_impl


def test_cli_and_orchestrator_draws_match(tmp_path: Path, monkeypatch) -> None:
    cfg_data = {
        "N_SIMULATIONS": 4,
        "N_MONTHS": 3,
        "financing_mode": "broadcast",
        "analysis_mode": "single_with_sensitivity",
        "w_beta_H": 0.6,
        "w_alpha_H": 0.4,
        "risk_metrics": ["terminal_ShortfallProb"],
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))

    idx_values = [0.01, 0.02, 0.015, 0.03, 0.005, 0.025]
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n" + "\n".join(str(val) for val in idx_values))
    idx_series = load_index_returns(idx_path)

    seed = 123
    orch_capture: dict[str, tuple[np.ndarray, ...]] = {}
    cli_capture: dict[str, tuple[np.ndarray, ...]] = {}
    orch_params: dict[str, object] = {}
    cli_params: dict[str, object] = {}
    run_phase = {"mode": "orch"}

    original_simulate_agents = sim_module.simulate_agents

    def capture_simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act):
        if run_phase["mode"] == "orch":
            if "draws" not in orch_capture:
                orch_capture["draws"] = (
                    np.array(r_beta),
                    np.array(r_H),
                    np.array(r_E),
                    np.array(r_M),
                )
        elif "draws" not in cli_capture:
            cli_capture["draws"] = (
                np.array(r_beta),
                np.array(r_H),
                np.array(r_E),
                np.array(r_M),
            )
        return original_simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)

    def capture_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        target = orch_params if run_phase["mode"] == "orch" else cli_params
        if "params" not in target:
            target["params"] = dict(params)
            if rng is not None:
                target["rng_state"] = rng.bit_generator.state
        return draw_joint_returns_impl(
            n_months=n_months, n_sim=n_sim, params=params, rng=rng, shocks=shocks
        )

    monkeypatch.setattr("pa_core.simulations.simulate_agents", capture_simulate_agents)
    monkeypatch.setattr("pa_core.sim.draw_joint_returns", capture_draw_joint_returns)

    cfg = load_config(cfg_path)
    orch = SimulatorOrchestrator(cfg, idx_series)
    orch.run(seed=seed)

    def noop_export(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("pa_core.reporting.export_to_excel", noop_export)
    run_phase["mode"] = "cli"

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
        ]
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
