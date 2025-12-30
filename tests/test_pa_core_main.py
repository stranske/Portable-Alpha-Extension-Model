from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

import pa_core.__main__ as pa_main
from pa_core.config import ModelConfig


def test_main_runs_with_return_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = ModelConfig(N_SIMULATIONS=2, N_MONTHS=3)
    monkeypatch.setattr(pa_main, "load_config", lambda _: cfg)

    backend_calls: Dict[str, Any] = {}

    def fake_resolve_and_set_backend(backend: str | None, _: ModelConfig) -> str:
        backend_calls["value"] = backend
        return "numpy"

    monkeypatch.setattr(
        pa_main, "resolve_and_set_backend", fake_resolve_and_set_backend
    )
    monkeypatch.setattr(
        pa_main,
        "select_vol_regime_sigma",
        lambda *_, **__: (0.12, None, None),
    )

    import pa_core.agents.registry as registry_mod
    import pa_core.data as data_mod
    import pa_core.random as random_mod
    import pa_core.reporting as reporting_mod
    import pa_core.sim as sim_mod
    import pa_core.sim.covariance as covariance_mod
    import pa_core.sim.metrics as metrics_mod
    import pa_core.simulations as simulations_mod

    monkeypatch.setattr(
        data_mod, "load_index_returns", lambda _: pd.Series([0.01, 0.02, -0.01])
    )
    monkeypatch.setattr(random_mod, "spawn_rngs", lambda *_: ["rng"])
    monkeypatch.setattr(
        random_mod,
        "spawn_agent_rngs",
        lambda *_: {"internal": "rng", "external_pa": "rng", "active_ext": "rng"},
    )
    monkeypatch.setattr(sim_mod, "draw_joint_returns", lambda **_: ("b", "h", "e", "m"))
    monkeypatch.setattr(
        sim_mod, "draw_financing_series", lambda **_: ("fi", "fe", "fa")
    )

    cov_calls: Dict[str, Any] = {}

    def fake_build_cov_matrix(*args: Any, **kwargs: Any) -> None:
        cov_calls["args"] = args
        cov_calls["kwargs"] = kwargs

    monkeypatch.setattr(covariance_mod, "build_cov_matrix", fake_build_cov_matrix)
    monkeypatch.setattr(registry_mod, "build_from_config", lambda _: ["agent"])
    monkeypatch.setattr(
        simulations_mod, "simulate_agents", lambda *_: {"Base": [1.0, 2.0]}
    )
    monkeypatch.setattr(
        metrics_mod, "summary_table", lambda *_, **__: {"ok": True}
    )

    export_calls: Dict[str, Any] = {}

    def fake_export_to_excel(
        inputs_dict: Dict[str, Any],
        summary: Dict[str, Any],
        raw_returns_dict: Dict[str, Any],
        filename: str,
    ) -> None:
        export_calls["inputs"] = inputs_dict
        export_calls["summary"] = summary
        export_calls["returns"] = raw_returns_dict
        export_calls["filename"] = filename

    monkeypatch.setattr(reporting_mod, "export_to_excel", fake_export_to_excel)

    output = tmp_path / "out.xlsx"
    pa_main.main(
        [
            "--config",
            "config.yml",
            "--index",
            "index.csv",
            "--output",
            str(output),
            "--backend",
            "numpy",
            "--seed",
            "42",
            "--return-distribution",
            "student_t",
            "--return-t-df",
            "7",
            "--return-copula",
            "t",
        ]
    )

    captured = capsys.readouterr()
    assert "[BACKEND] Using backend: numpy" in captured.out
    assert backend_calls["value"] == "numpy"
    assert cov_calls["kwargs"]["covariance_shrinkage"] == "none"
    assert cov_calls["kwargs"]["n_samples"] == 3
    assert export_calls["filename"] == str(output)
    assert export_calls["inputs"]["return_distribution"] == "student_t"
    assert export_calls["inputs"]["return_t_df"] == 7.0
    assert export_calls["inputs"]["return_copula"] == "t"


def test_main_rejects_invalid_vol_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConfig:
        vol_regime = "invalid"

        def model_dump(self) -> Dict[str, Any]:
            return {}

    monkeypatch.setattr(pa_main, "load_config", lambda _: FakeConfig())
    monkeypatch.setattr(pa_main, "resolve_and_set_backend", lambda *_: "numpy")
    import pa_core.data as data_mod
    import pa_core.random as random_mod

    monkeypatch.setattr(
        data_mod, "load_index_returns", lambda *_: pd.Series([0.01, 0.02])
    )
    monkeypatch.setattr(random_mod, "spawn_rngs", lambda *_: ["rng"])
    monkeypatch.setattr(
        random_mod,
        "spawn_agent_rngs",
        lambda *_: {"internal": "rng", "external_pa": "rng", "active_ext": "rng"},
    )

    with pytest.raises(ValueError, match="vol_regime must be 'single' or 'two_state'"):
        pa_main.main(["--config", "config.yml", "--index", "index.csv"])
