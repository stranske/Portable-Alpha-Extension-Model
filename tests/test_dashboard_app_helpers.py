from __future__ import annotations

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


def test_load_paths_sidecar_parquet_importerror_shows_hint(tmp_path, monkeypatch):
    xlsx = tmp_path / "Outputs.xlsx"
    xlsx.write_text("stub")

    parquet = tmp_path / "Outputs.parquet"
    parquet.write_text("stub")

    csv = tmp_path / "Outputs.csv"
    csv.write_text("a,b\n1,2\n")

    df = pd.DataFrame({"a": [1]})

    def _raise(*_args, **_kwargs):
        raise ImportError("no engine")

    seen = {}

    def _info(message):
        seen["message"] = message

    monkeypatch.setattr(pd, "read_parquet", _raise)
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(app.st, "info", _info)

    result = app._load_paths_sidecar(str(xlsx))

    assert result is df
    assert seen["message"] == app._PARQUET_HINT


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


def test_save_history_falls_back_to_csv_on_parquet_error(tmp_path, monkeypatch):
    df = pd.DataFrame({"Sim": [0, 1], "Return": [0.1, 0.2]})
    calls = []

    class _FakeStreamlit:
        def info(self, message: str) -> None:
            calls.append(message)

    def _raise_import_error(*_args, **_kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(df, "to_parquet", _raise_import_error)
    monkeypatch.setattr(app, "st", _FakeStreamlit())

    output_path = tmp_path / "history.parquet"
    app.save_history(df, base=output_path)

    assert calls
    assert output_path.with_suffix(".csv").exists()


def test_load_history_uses_csv_when_parquet_unavailable(tmp_path, monkeypatch):
    csv_path = tmp_path / "Outputs.csv"
    csv_path.write_text("Sim,Monthly_TR\n0,0.1\n0,0.3\n1,0.2\n")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("parquet read error")

    monkeypatch.setattr(pd, "read_parquet", _raise)

    result = app.load_history(parquet=str(tmp_path / "Outputs.parquet"))

    assert result is not None
    assert list(result.columns) == ["mean_return", "volatility"]
    assert result.loc[0, "mean_return"] == 0.2
