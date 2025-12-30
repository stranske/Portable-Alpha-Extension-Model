from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import pa_core.__main__ as pa_main


def _base_config_values() -> dict[str, float | int | str]:
    return {
        "mu_H": 0.12,
        "sigma_H": 0.2,
        "mu_E": 0.1,
        "sigma_E": 0.15,
        "mu_M": 0.08,
        "sigma_M": 0.18,
        "rho_idx_H": 0.4,
        "rho_idx_E": 0.3,
        "rho_idx_M": 0.2,
        "rho_H_E": 0.1,
        "rho_H_M": 0.05,
        "rho_E_M": 0.08,
        "return_distribution": "normal",
        "return_t_df": 6.0,
        "return_copula": "gaussian",
        "return_distribution_idx": "normal",
        "return_distribution_H": "normal",
        "return_distribution_E": "normal",
        "return_distribution_M": "normal",
        "internal_financing_mean_month": 0.01,
        "internal_financing_sigma_month": 0.02,
        "internal_spike_prob": 0.01,
        "internal_spike_factor": 2.0,
        "ext_pa_financing_mean_month": 0.015,
        "ext_pa_financing_sigma_month": 0.025,
        "ext_pa_spike_prob": 0.02,
        "ext_pa_spike_factor": 1.5,
        "act_ext_financing_mean_month": 0.012,
        "act_ext_financing_sigma_month": 0.022,
        "act_ext_spike_prob": 0.03,
        "act_ext_spike_factor": 1.2,
        "N_SIMULATIONS": 2,
        "N_MONTHS": 3,
    }


@dataclass
class DataclassConfig:
    mu_H: float
    sigma_H: float
    mu_E: float
    sigma_E: float
    mu_M: float
    sigma_M: float
    rho_idx_H: float
    rho_idx_E: float
    rho_idx_M: float
    rho_H_E: float
    rho_H_M: float
    rho_E_M: float
    return_distribution: str
    return_t_df: float
    return_copula: str
    return_distribution_idx: str
    return_distribution_H: str
    return_distribution_E: str
    return_distribution_M: str
    internal_financing_mean_month: float
    internal_financing_sigma_month: float
    internal_spike_prob: float
    internal_spike_factor: float
    ext_pa_financing_mean_month: float
    ext_pa_financing_sigma_month: float
    ext_pa_spike_prob: float
    ext_pa_spike_factor: float
    act_ext_financing_mean_month: float
    act_ext_financing_sigma_month: float
    act_ext_spike_prob: float
    act_ext_spike_factor: float
    N_SIMULATIONS: int
    N_MONTHS: int

    last_payload: dict[str, float | int | str] | None = None

    @classmethod
    def model_validate(cls, payload: dict[str, float | int | str]) -> "DataclassConfig":
        cls.last_payload = payload
        return cls(**payload)

    def model_dump(self) -> dict[str, float | int | str]:
        return _base_config_values()


class SimpleConfig:
    last_payload: dict[str, float | int | str] | None = None

    def __init__(self, values: dict[str, float | int | str]) -> None:
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    @classmethod
    def model_validate(cls, payload: dict[str, float | int | str]) -> "SimpleConfig":
        cls.last_payload = payload
        return cls(payload)

    def model_dump(self) -> dict[str, float | int | str]:
        return self._values


def _setup_common_stubs(monkeypatch):
    monkeypatch.setattr(
        pa_main, "resolve_and_set_backend", lambda backend, cfg: "numpy"
    )

    import pa_core.agents.registry as registry
    import pa_core.data as data
    import pa_core.random as random
    import pa_core.reporting as reporting
    import pa_core.sim as sim
    import pa_core.sim.covariance as covariance
    import pa_core.sim.metrics as metrics
    import pa_core.simulations as simulations

    monkeypatch.setattr(
        data, "load_index_returns", lambda path: pd.Series([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(random, "spawn_rngs", lambda seed, n: ["rng"] * n)
    monkeypatch.setattr(
        random,
        "spawn_agent_rngs",
        lambda seed, names: {name: f"rng-{name}" for name in names},
    )
    monkeypatch.setattr(covariance, "build_cov_matrix", lambda *args, **kwargs: "cov")
    monkeypatch.setattr(registry, "build_from_config", lambda cfg: ["agent"])
    monkeypatch.setattr(
        sim, "draw_joint_returns", lambda **kwargs: (["b"], ["h"], ["e"], ["m"])
    )
    monkeypatch.setattr(
        sim, "draw_financing_series", lambda **kwargs: (["i"], ["e"], ["a"])
    )
    monkeypatch.setattr(
        simulations, "simulate_agents", lambda *args, **kwargs: {"Base": [1, 2]}
    )
    monkeypatch.setattr(metrics, "summary_table", lambda *args, **kwargs: "summary")
    monkeypatch.setattr(reporting, "export_to_excel", lambda *args, **kwargs: None)


def test_main_applies_return_overrides(monkeypatch):
    config_values = _base_config_values()
    config = DataclassConfig(**config_values)

    monkeypatch.setattr(pa_main, "load_config", lambda path: config)
    _setup_common_stubs(monkeypatch)

    import pa_core.reporting as reporting

    captured = {}

    def _export(inputs_dict, summary, raw_returns, filename):
        captured["inputs"] = inputs_dict
        captured["summary"] = summary
        captured["raw_returns"] = raw_returns
        captured["filename"] = filename

    monkeypatch.setattr(reporting, "export_to_excel", _export)

    pa_main.main(
        [
            "--config",
            "config.yml",
            "--index",
            "index.csv",
            "--output",
            "out.xlsx",
            "--return-distribution",
            "student_t",
            "--return-t-df",
            "4",
            "--return-copula",
            "t",
        ]
    )

    assert captured["filename"] == "out.xlsx"
    assert DataclassConfig.last_payload is not None
    assert DataclassConfig.last_payload["return_distribution"] == "student_t"
    assert DataclassConfig.last_payload["return_t_df"] == 4.0
    assert DataclassConfig.last_payload["return_copula"] == "t"


def test_main_runs_without_overrides(monkeypatch):
    config_values = _base_config_values()
    config = SimpleConfig(config_values)

    monkeypatch.setattr(pa_main, "load_config", lambda path: config)
    _setup_common_stubs(monkeypatch)

    import pa_core.reporting as reporting

    called = {}

    def _export(inputs_dict, summary, raw_returns, filename):
        called["filename"] = filename

    monkeypatch.setattr(reporting, "export_to_excel", _export)

    pa_main.main(["--config", "config.yml", "--index", "index.csv"])

    assert called["filename"] == "Outputs.xlsx"


def test_main_applies_overrides_for_model_dump_config(monkeypatch):
    config_values = _base_config_values()
    config = SimpleConfig(config_values)

    monkeypatch.setattr(pa_main, "load_config", lambda path: config)
    _setup_common_stubs(monkeypatch)

    pa_main.main(
        [
            "--config",
            "config.yml",
            "--index",
            "index.csv",
            "--return-distribution",
            "student_t",
            "--return-t-df",
            "5",
        ]
    )

    assert SimpleConfig.last_payload is not None
    assert SimpleConfig.last_payload["return_distribution"] == "student_t"
    assert SimpleConfig.last_payload["return_t_df"] == 5.0
