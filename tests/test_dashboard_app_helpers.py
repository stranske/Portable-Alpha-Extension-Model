from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from dashboard import app
from pa_core.contracts import SUMMARY_SHEET_NAME


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


def test_load_data_workbook_revision_invalidates_same_path_replacement(tmp_path) -> None:
    xlsx = tmp_path / "Outputs.xlsx"
    pd.DataFrame({"monthly_TE": [0.01]}).to_excel(xlsx, sheet_name=SUMMARY_SHEET_NAME, index=False)
    revision_a = app.workbook_cache_revision(str(xlsx))
    summary_a, _ = app.load_data(str(xlsx), revision_a)
    assert float(summary_a["monthly_TE"].iloc[0]) == pytest.approx(0.01)

    pd.DataFrame({"monthly_TE": [0.88]}).to_excel(xlsx, sheet_name=SUMMARY_SHEET_NAME, index=False)
    revision_b = app.workbook_cache_revision(str(xlsx))
    assert revision_b != revision_a
    summary_b, _ = app.load_data(str(xlsx), revision_b)
    assert float(summary_b["monthly_TE"].iloc[0]) == pytest.approx(0.88)

    frozen_stat = xlsx.stat()
    pd.DataFrame({"monthly_TE": [0.42]}).to_excel(xlsx, sheet_name=SUMMARY_SHEET_NAME, index=False)
    os.utime(xlsx, ns=(frozen_stat.st_atime_ns, frozen_stat.st_mtime_ns))
    with patch("dashboard.app.Path.stat", return_value=frozen_stat):
        revision_same_stat = app.workbook_cache_revision(str(xlsx))
    assert revision_same_stat != revision_b
    summary_same_stat, _ = app.load_data(str(xlsx), revision_same_stat)
    assert float(summary_same_stat["monthly_TE"].iloc[0]) == pytest.approx(0.42)


def test_workbook_replacement_during_summary_read_keeps_revision_and_data_aligned(
    tmp_path, monkeypatch
) -> None:
    xlsx = tmp_path / "Outputs.xlsx"
    replacement = tmp_path / "Replacement.xlsx"
    pd.DataFrame({"monthly_TE": [0.01]}).to_excel(xlsx, sheet_name=SUMMARY_SHEET_NAME, index=False)
    pd.DataFrame({"monthly_TE": [0.88]}).to_excel(
        replacement, sheet_name=SUMMARY_SHEET_NAME, index=False
    )
    original_bytes = xlsx.read_bytes()
    original_stat = xlsx.stat()
    original_revision = app.workbook_cache_revision(str(xlsx))
    read_excel = pd.read_excel
    summary_reads = 0

    def read_with_replacement(*args, **kwargs):
        nonlocal summary_reads
        frame = read_excel(*args, **kwargs)
        if kwargs.get("sheet_name") == SUMMARY_SHEET_NAME:
            summary_reads += 1
            if summary_reads == 1:
                replacement.replace(xlsx)
        return frame

    monkeypatch.setattr(pd, "read_excel", read_with_replacement)
    summary, _ = app.load_results_workbook(str(xlsx))

    assert float(summary["monthly_TE"].iloc[0]) == pytest.approx(0.01)
    assert summary_reads == 1
    assert app.workbook_cache_revision(str(xlsx)) != original_revision
    replaced_summary, _ = app.load_results_workbook(str(xlsx))
    assert float(replaced_summary["monthly_TE"].iloc[0]) == pytest.approx(0.88)

    xlsx.write_bytes(original_bytes)
    os.utime(xlsx, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert app.workbook_cache_revision(str(xlsx)) == original_revision
    restored_summary, _ = app.load_results_workbook(str(xlsx))
    pd.testing.assert_frame_equal(restored_summary, summary)


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
    csv_path.write_text("Sim,Return\n0,0.1\n0,0.3\n1,0.2\n")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("parquet read error")

    monkeypatch.setattr(pd, "read_parquet", _raise)

    result = app.load_history(parquet=str(tmp_path / "Outputs.parquet"))

    assert result is not None
    assert list(result.columns) == ["mean_return", "volatility"]
    assert result.loc[0, "mean_return"] == 0.2
