from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

import pa_core.__main__ as pa_main
import pa_core.agents.registry as registry
import pa_core.data as data_module
import pa_core.random as random_module
import pa_core.reporting as reporting_module
import pa_core.sim as sim_module
import pa_core.sim.covariance as covariance_module
import pa_core.sim.metrics as metrics_module
import pa_core.simulations as simulations_module


class FakeConfig:
    def __init__(self, **data: Any) -> None:
        self._data = dict(data)
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "FakeConfig":
        return cls(**data)


def _base_config_data() -> Dict[str, Any]:
    return {
        "N_SIMULATIONS": 2,
        "N_MONTHS": 3,
        "mu_H": 0.04,
        "sigma_H": 0.01,
        "mu_E": 0.05,
        "sigma_E": 0.02,
        "mu_M": 0.03,
        "sigma_M": 0.02,
        "rho_idx_H": 0.05,
        "rho_idx_E": 0.0,
        "rho_idx_M": 0.0,
        "rho_H_E": 0.1,
        "rho_H_M": 0.1,
        "rho_E_M": 0.0,
        "return_distribution": "normal",
        "return_t_df": 5.0,
        "return_copula": "gaussian",
        "return_distribution_idx": None,
        "return_distribution_H": None,
        "return_distribution_E": None,
        "return_distribution_M": None,
        "internal_financing_mean_month": 0.0,
        "internal_financing_sigma_month": 0.0,
        "internal_spike_prob": 0.0,
        "internal_spike_factor": 0.0,
        "ext_pa_financing_mean_month": 0.0,
        "ext_pa_financing_sigma_month": 0.0,
        "ext_pa_spike_prob": 0.0,
        "ext_pa_spike_factor": 0.0,
        "act_ext_financing_mean_month": 0.0,
        "act_ext_financing_sigma_month": 0.0,
        "act_ext_spike_prob": 0.0,
        "act_ext_spike_factor": 0.0,
    }


def _patch_main_dependencies(monkeypatch):
    calls: Dict[str, Any] = {}

    def fake_load_index_returns(path: str):
        calls["index_path"] = path
        return pd.Series([0.1, 0.2, 0.3])

    def fake_spawn_rngs(seed: int | None, count: int):
        calls["spawn_rngs"] = (seed, count)
        return ["rng"]

    def fake_spawn_agent_rngs_with_ids(seed: int | None, names, **_kwargs):
        calls["spawn_agent_rngs_with_ids"] = (seed, list(names))
        return {"internal": "rng-int"}, {"internal": "substream"}

    def fake_build_cov_matrix(*args, **kwargs):
        calls["cov_matrix"] = {"args": args, "kwargs": kwargs}
        return "cov"

    def fake_draw_joint_returns(*args, **kwargs):
        calls["draw_joint_returns"] = {"args": args, "kwargs": kwargs}
        return ("beta", "H", "E", "M")

    def fake_draw_financing_series(*args, **kwargs):
        calls["draw_financing_series"] = {"args": args, "kwargs": kwargs}
        return ("int", "ext", "act")

    def fake_build_from_config(config):
        calls["build_from_config"] = config
        return "agents"

    def fake_simulate_agents(*args, **kwargs):
        calls["simulate_agents"] = {"args": args, "kwargs": kwargs}
        return {"Base": [[1.0, 2.0], [3.0, 4.0]]}

    def fake_summary_table(returns, benchmark):
        calls["summary_table"] = (returns, benchmark)
        return "summary"

    def fake_export_to_excel(inputs_dict, summary, raw_returns_dict, filename, **_kwargs):
        calls["export_to_excel"] = {
            "inputs": inputs_dict,
            "summary": summary,
            "raw_returns": raw_returns_dict,
            "filename": filename,
        }

    monkeypatch.setattr(data_module, "load_index_returns", fake_load_index_returns)
    monkeypatch.setattr(random_module, "spawn_rngs", fake_spawn_rngs)
    monkeypatch.setattr(random_module, "spawn_agent_rngs_with_ids", fake_spawn_agent_rngs_with_ids)
    monkeypatch.setattr(covariance_module, "build_cov_matrix", fake_build_cov_matrix)
    monkeypatch.setattr(sim_module, "draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr(sim_module, "draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr(registry, "build_from_config", fake_build_from_config)
    monkeypatch.setattr(simulations_module, "simulate_agents", fake_simulate_agents)
    monkeypatch.setattr(metrics_module, "summary_table", fake_summary_table)
    monkeypatch.setattr(reporting_module, "export_to_excel", fake_export_to_excel)

    return calls


def test_main_applies_overrides_and_exports(monkeypatch) -> None:
    cfg = FakeConfig(**_base_config_data())
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    def fake_resolve_and_set_backend(choice, config):
        calls["resolve_backend"] = (choice, config)
        return "numpy"

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)
    monkeypatch.setattr(pa_main, "resolve_and_set_backend", fake_resolve_and_set_backend)

    pa_main.main(
        [
            "--config",
            "config.yaml",
            "--index",
            "index.csv",
            "--output",
            "out.xlsx",
            "--backend",
            "numpy",
            "--seed",
            "123",
            "--return-distribution",
            "student_t",
            "--return-t-df",
            "7",
            "--return-copula",
            "t",
        ]
    )

    assert calls["load_config"] == "config.yaml"
    assert calls["resolve_backend"][0] == "numpy"
    assert calls["build_from_config"].return_distribution == "student_t"
    assert calls["build_from_config"].return_t_df == 7.0
    assert calls["build_from_config"].return_copula == "t"
    export_call = calls["export_to_excel"]
    assert export_call["filename"] == "out.xlsx"
    assert export_call["inputs"]["return_distribution"] == "student_t"
    assert export_call["inputs"]["return_t_df"] == 7.0
    assert export_call["inputs"]["return_copula"] == "t"
    assert isinstance(export_call["raw_returns"]["Base"], pd.DataFrame)


def test_main_without_overrides(monkeypatch) -> None:
    cfg = FakeConfig(**_base_config_data())
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    def fake_resolve_and_set_backend(choice, config):
        calls["resolve_backend"] = (choice, config)
        return "numpy"

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)
    monkeypatch.setattr(pa_main, "resolve_and_set_backend", fake_resolve_and_set_backend)

    pa_main.main(["--config", "config.yaml", "--index", "index.csv"])

    assert calls["build_from_config"] is cfg


def test_main_rejects_invalid_vol_regime(monkeypatch) -> None:
    cfg = FakeConfig(**{**_base_config_data(), "vol_regime": "invalid"})
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    def fake_resolve_and_set_backend(choice, config):
        calls["resolve_backend"] = (choice, config)
        return "numpy"

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)
    monkeypatch.setattr(pa_main, "resolve_and_set_backend", fake_resolve_and_set_backend)

    with pytest.raises(ValueError, match="vol_regime must be 'single' or 'two_state'"):
        pa_main.main(["--config", "config.yaml", "--index", "index.csv"])
