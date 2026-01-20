from __future__ import annotations

import pandas as pd

from pa_core.cli import main
from pa_core.config import ModelConfig


def _mock_monthly_series() -> pd.Series:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    series = pd.Series([0.01, 0.02], index=dates)
    series.attrs["frequency"] = "monthly"
    return series


def test_cli_validate_only_skips_run(monkeypatch, capsys) -> None:
    cfg = ModelConfig(N_SIMULATIONS=100, N_MONTHS=1, financing_mode="broadcast")
    monkeypatch.setattr("pa_core.config.load_config", lambda *_: cfg)
    monkeypatch.setattr("pa_core.backend.resolve_and_set_backend", lambda *_: "numpy")
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda *_: _mock_monthly_series())
    monkeypatch.setattr("pa_core.units.normalize_index_series", lambda series, *_: series)

    called = {"run_single": False}

    def _fail_run_single(*_args, **_kwargs) -> None:
        called["run_single"] = True
        raise AssertionError("run_single should not be called when --validate-only is set")

    monkeypatch.setattr("pa_core.facade.run_single", _fail_run_single)

    main(["--config", "cfg.yaml", "--index", "idx.csv", "--validate-only"])
    captured = capsys.readouterr()

    assert called["run_single"] is False
    assert "Validation completed successfully." in captured.out
