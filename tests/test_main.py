from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, get_origin

import numpy as np
import pandas as pd

import pa_core.__main__ as pa_main


@dataclass
class DummyConfig:
    mu_H: float = 0.06
    sigma_H: float = 0.12
    mu_E: float = 0.07
    sigma_E: float = 0.15
    mu_M: float = 0.05
    sigma_M: float = 0.1
    rho_idx_H: float = 0.1
    rho_idx_E: float = 0.2
    rho_idx_M: float = 0.3
    rho_H_E: float = 0.4
    rho_H_M: float = 0.5
    rho_E_M: float = 0.6
    correlation_repair_mode: str = "warn_fix"
    correlation_repair_shrinkage: float = 0.0
    return_distribution: str = "normal"
    return_t_df: float = 8.0
    return_copula: str = "gaussian"
    return_distribution_idx: str = "normal"
    return_distribution_H: str = "normal"
    return_distribution_E: str = "normal"
    return_distribution_M: str = "normal"
    internal_financing_mean_month: float = 0.0
    internal_financing_sigma_month: float = 0.01
    internal_spike_prob: float = 0.0
    internal_spike_factor: float = 1.0
    ext_pa_financing_mean_month: float = 0.0
    ext_pa_financing_sigma_month: float = 0.01
    ext_pa_spike_prob: float = 0.0
    ext_pa_spike_factor: float = 1.0
    act_ext_financing_mean_month: float = 0.0
    act_ext_financing_sigma_month: float = 0.01
    act_ext_spike_prob: float = 0.0
    act_ext_spike_factor: float = 1.0
    financing_mode: str = "broadcast"
    vol_regime: str = "single"
    vol_regime_window: int = 12
    covariance_shrinkage: str = "none"
    regimes: Any = None
    regime_start: Any = None
    regime_transition: Any = None
    N_SIMULATIONS: int = 2
    N_MONTHS: int = 3

    last_validated: ClassVar[Dict[str, Any] | None] = None

    def model_dump(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "DummyConfig":
        cls.last_validated = dict(data)
        # Filter out ClassVar fields that shouldn't be passed to __init__
        init_data = {
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
            and get_origin(cls.__dataclass_fields__[k].type) is not ClassVar
        }
        return cls(**init_data)


def test_main_applies_overrides_and_exports(monkeypatch, tmp_path) -> None:
    export_calls: Dict[str, Any] = {}

    def fake_load_config(_: str) -> DummyConfig:
        return DummyConfig()

    def fake_build_from_config(cfg: DummyConfig) -> Dict[str, str]:
        return {"agent": "stub"}

    def fake_load_index_returns(_: str) -> pd.Series:
        return pd.Series([0.01, 0.02, 0.03])

    def fake_spawn_rngs(seed: int | None, n: int) -> list[object]:
        return [object() for _ in range(n)]

    def fake_spawn_agent_rngs_with_ids(
        seed: int | None, names: list[str], **_kwargs
    ) -> tuple[dict[str, object], dict[str, str]]:
        return {name: object() for name in names}, {name: "substream" for name in names}

    def fake_draw_joint_returns(
        **_: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.zeros((2, 3))
        return arr, arr, arr, arr

    def fake_draw_financing_series(
        **_: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.zeros((2, 3))
        return arr, arr, arr

    def fake_build_cov_matrix(*_: Any, **__: Any) -> None:
        return None

    def fake_simulate_agents(*_: Any, **__: Any) -> Dict[str, np.ndarray]:
        return {"Base": np.array([[0.01, 0.02], [0.03, 0.04]])}

    def fake_summary_table(returns: Dict[str, np.ndarray], benchmark: str) -> str:
        assert benchmark == "Base"
        assert "Base" in returns
        return "summary"

    def fake_export_to_excel(
        inputs: Dict[str, Any],
        summary: str,
        raw_returns: Dict[str, pd.DataFrame],
        filename: str,
        **_kwargs: Any,
    ) -> None:
        export_calls["inputs"] = inputs
        export_calls["summary"] = summary
        export_calls["raw_returns"] = raw_returns
        export_calls["filename"] = filename

    monkeypatch.setattr(pa_main, "load_config", fake_load_config)
    monkeypatch.setattr("pa_core.agents.registry.build_from_config", fake_build_from_config)
    monkeypatch.setattr("pa_core.data.load_index_returns", fake_load_index_returns)
    monkeypatch.setattr("pa_core.random.spawn_rngs", fake_spawn_rngs)
    monkeypatch.setattr("pa_core.random.spawn_agent_rngs_with_ids", fake_spawn_agent_rngs_with_ids)
    monkeypatch.setattr("pa_core.sim.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.sim.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sim.covariance.build_cov_matrix", fake_build_cov_matrix)
    monkeypatch.setattr("pa_core.simulations.simulate_agents", fake_simulate_agents)
    monkeypatch.setattr("pa_core.sim.metrics.summary_table", fake_summary_table)
    monkeypatch.setattr("pa_core.reporting.export_to_excel", fake_export_to_excel)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("stub: true\n")
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("date,return\n2020-01-01,0.01\n")

    pa_main.main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_path),
            "--output",
            "Outputs.xlsx",
            "--backend",
            "numpy",
            "--seed",
            "123",
            "--return-distribution",
            "student_t",
            "--return-t-df",
            "5.5",
            "--return-copula",
            "t",
        ]
    )

    assert DummyConfig.last_validated is not None
    assert DummyConfig.last_validated["return_distribution"] == "student_t"
    assert DummyConfig.last_validated["return_t_df"] == 5.5
    assert DummyConfig.last_validated["return_copula"] == "t"
    assert export_calls["summary"] == "summary"
    assert export_calls["filename"] == "Outputs.xlsx"
    assert export_calls["inputs"]["return_distribution"] == "student_t"
    assert "Base" in export_calls["raw_returns"]
