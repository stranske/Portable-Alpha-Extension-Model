from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard import app


def test_load_paths_sidecar_csv_fallback(tmp_path, monkeypatch):
    xlsx = tmp_path / "Outputs.xlsx"
    xlsx.write_text("stub")

    parquet = tmp_path / "Outputs.parquet"
    parquet.write_text("stub")

    csv = tmp_path / "Outputs.csv"
    csv.write_text("a,b\n1,2\n")

    df = pd.DataFrame({"a": [1]})

    def _raise(*_args, **_kwargs):
        raise RuntimeError("parquet failure")

    monkeypatch.setattr(pd, "read_parquet", _raise)
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: df)

    result = app._load_paths_sidecar(str(xlsx))

    assert result is df


def test_load_paths_sidecar_excel_fallback(tmp_path, monkeypatch):
    xlsx = tmp_path / "Outputs.xlsx"
    xlsx.write_text("stub")

    df = pd.DataFrame({"Sim": [0], "Return": [0.1]})
    monkeypatch.setattr(pd, "read_excel", lambda *_args, **_kwargs: df)

    result = app._load_paths_sidecar(str(xlsx))

    assert result is df


def test_get_plot_fn_returns_callable():
    plot_fn = app._get_plot_fn("pa_core.viz.risk_return.make")

    assert callable(plot_fn)
