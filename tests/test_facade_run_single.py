import pandas as pd

from pa_core.config import RegimeConfig, load_config
from pa_core.facade import RunOptions, run_single


def test_run_single_returns_artifacts() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    artifacts = run_single(cfg, idx, RunOptions(seed=123))

    assert artifacts.summary is not None
    assert not artifacts.summary.empty
    assert artifacts.config.N_SIMULATIONS == 2
    assert set(artifacts.returns).issubset(set(artifacts.raw_returns))
    assert artifacts.manifest is not None
    assert artifacts.manifest["seed"] == 123
    assert set(artifacts.manifest["substream_ids"]) == {
        "internal",
        "external_pa",
        "active_ext",
    }


def test_run_single_applies_config_overrides() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    artifacts = run_single(cfg, idx, RunOptions(seed=123, config_overrides={"N_SIMULATIONS": 1}))

    assert artifacts.config.N_SIMULATIONS == 1


def test_run_single_uses_regime_paths(monkeypatch) -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 4,
            "N_MONTHS": 6,
            "regimes": [
                RegimeConfig(name="calm"),
                RegimeConfig(
                    name="stress",
                    idx_sigma_multiplier=2.0,
                    sigma_H=0.06,
                    sigma_E=0.06,
                    sigma_M=0.06,
                    rho_idx_H=0.8,
                    rho_idx_E=0.8,
                    rho_idx_M=0.8,
                    rho_H_E=0.85,
                    rho_H_M=0.85,
                    rho_E_M=0.85,
                ),
            ],
            "regime_transition": [[0.9, 0.1], [0.2, 0.8]],
            "regime_start": "calm",
        }
    )
    idx = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01])

    from pa_core import sim as sim_module

    called: dict[str, object] = {}
    original = sim_module.draw_joint_returns

    def _spy_draw_joint_returns(*args: object, **kwargs: object):
        called["regime_paths"] = kwargs.get("regime_paths")
        called["regime_params"] = kwargs.get("regime_params")
        return original(*args, **kwargs)

    monkeypatch.setattr(sim_module, "draw_joint_returns", _spy_draw_joint_returns)

    artifacts = run_single(cfg, idx, RunOptions(seed=123))

    assert called["regime_paths"] is not None
    assert called["regime_params"] is not None
    assert "Regime" in artifacts.raw_returns
    regime_values = set(pd.unique(artifacts.raw_returns["Regime"].to_numpy().ravel()))
    assert regime_values.issubset({"calm", "stress"})


def test_run_single_attaches_correlation_repair_frames() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 2,
            "N_MONTHS": 3,
            "rho_idx_H": 0.9,
            "rho_idx_E": 0.9,
            "rho_idx_M": 0.0,
            "rho_H_E": -0.9,
            "rho_H_M": 0.0,
            "rho_E_M": 0.0,
            "correlation_repair_mode": "warn_fix",
            "correlation_repair_shrinkage": 0.1,
        }
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    artifacts = run_single(cfg, idx, RunOptions(seed=123))

    for key in ("_corr_before_df", "_corr_after_df", "_corr_delta_df", "_corr_repair_info_df"):
        assert key in artifacts.inputs
        assert isinstance(artifacts.inputs[key], pd.DataFrame)
    assert artifacts.inputs["_corr_before_df"].shape == (4, 4)
    assert artifacts.inputs["_corr_after_df"].shape == (4, 4)
