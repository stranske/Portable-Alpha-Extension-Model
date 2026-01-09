from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import pa_core.__main__ as pa_module
import pa_core.cli as pa_cli
import pa_core.pa as pa_entry
from pa_core.facade import RunArtifacts


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
    monkeypatch.setattr("pa_core.reporting.constraints.build_constraint_report", lambda *_: pd.DataFrame())
    monkeypatch.setattr("pa_core.reporting.console.print_constraint_report", lambda *_: None)


def test_non_canonical_cli_emits_warning_without_output(monkeypatch, capsys, tmp_path) -> None:
    cfg = DummyConfig()
    _patch_cli_run(monkeypatch, cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pa_cli.main(
            [
                "--config",
                "cfg.yaml",
                "--index",
                "idx.csv",
                "--output",
                str(tmp_path / "out.xlsx"),
            ],
            emit_deprecation_warning=True,
        )

    captured = capsys.readouterr()
    assert any(isinstance(entry.message, DeprecationWarning) for entry in caught)
    assert "deprecated" not in captured.out.lower()
    assert "deprecated" not in captured.err.lower()


def test_non_canonical_module_emits_warning_without_output(monkeypatch, capsys) -> None:
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pa_module.main(
            [
                "--config",
                "config.yaml",
                "--index",
                "index.csv",
                "--output",
                "out.xlsx",
            ]
        )

    captured = capsys.readouterr()
    assert any(isinstance(entry.message, DeprecationWarning) for entry in caught)
    assert "deprecated" not in captured.out.lower()
    assert "deprecated" not in captured.err.lower()


def test_canonical_pa_run_has_no_warning(monkeypatch, capsys, tmp_path) -> None:
    cfg = DummyConfig()
    _patch_cli_run(monkeypatch, cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pa_entry.main(
            [
                "run",
                "--config",
                "cfg.yaml",
                "--index",
                "idx.csv",
                "--output",
                str(tmp_path / "out.xlsx"),
            ]
        )

    captured = capsys.readouterr()
    assert not any(isinstance(entry.message, DeprecationWarning) for entry in caught)
    assert "deprecated" not in captured.out.lower()
    assert "deprecated" not in captured.err.lower()
