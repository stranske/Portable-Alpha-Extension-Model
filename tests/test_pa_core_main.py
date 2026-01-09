from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

import pa_core.__main__ as pa_main


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
        "financing_mode": "broadcast",
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
        "correlation_repair_mode": "warn_fix",
        "correlation_repair_shrinkage": 0.0,
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
        "regimes": None,
        "regime_start": None,
        "regime_transition": None,
    }


def _patch_main_dependencies(monkeypatch):
    calls: Dict[str, Any] = {}

    def fake_load_index_returns(path: str):
        calls["index_path"] = path
        return pd.Series([0.1, 0.2, 0.3])

    def fake_run_single(config, index_series, options):
        calls["run_single"] = {
            "config": config,
            "index_series": index_series,
            "options": options,
        }
        return object()

    def fake_export(artifacts, output_path, **_kwargs):
        calls["export"] = {"artifacts": artifacts, "output_path": output_path}

    def fake_get_backend():
        calls["get_backend"] = True
        return "numpy"

    monkeypatch.setattr(pa_main, "load_index_returns", fake_load_index_returns)
    monkeypatch.setattr(pa_main, "run_single", fake_run_single)
    monkeypatch.setattr(pa_main, "export", fake_export)
    monkeypatch.setattr(pa_main, "get_backend", fake_get_backend)

    return calls


def test_main_applies_overrides_and_exports(monkeypatch) -> None:
    cfg = FakeConfig(**_base_config_data())
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)

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
    run_call = calls["run_single"]
    # Options should contain overrides; config passed to run_single is original
    assert run_call["config"] is cfg
    assert run_call["options"].return_distribution == "student_t"
    assert run_call["options"].return_t_df == 7.0
    assert run_call["options"].return_copula == "t"
    assert run_call["options"].seed == 123
    assert run_call["options"].backend == "numpy"
    assert calls["export"]["output_path"] == "out.xlsx"


def test_main_without_overrides(monkeypatch) -> None:
    cfg = FakeConfig(**_base_config_data())
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)

    pa_main.main(["--config", "config.yaml", "--index", "index.csv"])

    assert calls["run_single"]["config"] is cfg


def test_main_rejects_invalid_vol_regime(monkeypatch) -> None:
    cfg = FakeConfig(**{**_base_config_data(), "vol_regime": "invalid"})
    calls = _patch_main_dependencies(monkeypatch)

    def fake_load_config(path: str):
        calls["load_config"] = path
        return cfg

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)

    with pytest.raises(ValueError, match="vol_regime must be 'single' or 'two_state'"):
        pa_main.main(["--config", "config.yaml", "--index", "index.csv"])
