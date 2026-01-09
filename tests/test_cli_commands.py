from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

import pa_core.__main__ as pa_module
import pa_core.pa as pa_entry
from pa_core.facade import RunArtifacts

EXPECTED_BACKEND_STDOUT = "[BACKEND] Using backend: numpy\n"
EXPECTED_STDERR_EMPTY = ""


@dataclass
class DummyConfig:
    analysis_mode: str = "single_with_sensitivity"

    def model_dump(self) -> dict[str, Any]:
        return {"analysis_mode": self.analysis_mode}


def _make_index_series() -> pd.Series:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    series = pd.Series([0.01, 0.02], index=dates)
    series.attrs["frequency"] = "monthly"
    return series


def _fake_run_artifacts(cfg: DummyConfig) -> RunArtifacts:
    summary = pd.DataFrame({"Agent": ["Base"]})
    returns = {"Base": np.array([[0.01, 0.02]])}
    raw_returns = {"Base": pd.DataFrame({"returns": [0.01, 0.02]})}
    return RunArtifacts(
        config=cfg,
        index_series=_make_index_series(),
        returns=returns,
        summary=summary,
        inputs={},
        raw_returns=raw_returns,
        manifest={},
    )


def _exit_code(call: Callable[[], None]) -> int:
    try:
        call()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


def _patch_cli_run(monkeypatch, cfg: DummyConfig) -> None:
    monkeypatch.setattr("pa_core.config.load_config", lambda *_: cfg)
    monkeypatch.setattr("pa_core.facade.apply_run_options", lambda *_: cfg)
    monkeypatch.setattr("pa_core.backend.resolve_and_set_backend", lambda *_: "numpy")
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda *_: _make_index_series())
    monkeypatch.setattr("pa_core.units.normalize_index_series", lambda series, *_: series)
    monkeypatch.setattr("pa_core.facade.run_single", lambda *_: _fake_run_artifacts(cfg))
    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_: None)
    monkeypatch.setattr("pa_core.reporting.export_to_excel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pa_core.reporting.attribution.compute_sleeve_return_attribution",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "pa_core.reporting.attribution.compute_sleeve_return_contribution",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "pa_core.reporting.attribution.compute_sleeve_cvar_contribution",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "pa_core.reporting.attribution.compute_sleeve_risk_attribution",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "pa_core.reporting.constraints.build_constraint_report", lambda *_: pd.DataFrame()
    )
    monkeypatch.setattr("pa_core.reporting.console.print_constraint_report", lambda *_: None)


def test_pa_run_command_outputs_and_exit_code(monkeypatch, tmp_path, capsys) -> None:
    cfg = DummyConfig()
    _patch_cli_run(monkeypatch, cfg)

    out_path = tmp_path / "out.xlsx"
    exit_code = _exit_code(
        lambda: pa_entry.main(
            [
                "run",
                "--config",
                "cfg.yaml",
                "--index",
                "idx.csv",
                "--output",
                str(out_path),
            ]
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == EXPECTED_BACKEND_STDOUT
    assert captured.err == EXPECTED_STDERR_EMPTY


def test_module_command_outputs_and_exit_code(monkeypatch, capsys) -> None:
    def fake_load_config(_: str) -> DummyConfig:
        return DummyConfig()

    def fake_run_single(*_args: Any, **_kwargs: Any) -> object:
        return object()

    def fake_export(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pa_module, "load_config", fake_load_config)
    monkeypatch.setattr(pa_module, "load_index_returns", lambda *_: _make_index_series())
    monkeypatch.setattr(pa_module, "normalize_index_series", lambda series, *_: series)
    monkeypatch.setattr(
        pa_module, "select_vol_regime_sigma", lambda *_args, **_kwargs: (0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(pa_module, "run_single", fake_run_single)
    monkeypatch.setattr(pa_module, "export", fake_export)
    monkeypatch.setattr(pa_module, "get_backend", lambda: "numpy")

    exit_code = _exit_code(
        lambda: pa_module.main(
            [
                "--config",
                "config.yaml",
                "--index",
                "index.csv",
                "--output",
                "out.xlsx",
            ]
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == EXPECTED_BACKEND_STDOUT
    assert captured.err == EXPECTED_STDERR_EMPTY
