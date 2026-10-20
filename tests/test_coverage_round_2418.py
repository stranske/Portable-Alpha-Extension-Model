"""Focused coverage for the ordered targets in issue #2418."""

from __future__ import annotations

import asyncio
import io
import runpy
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest
from PIL import Image


def test_pdf_report_export_and_fallback_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pa_core.viz import pdf_report

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    fig = go.Figure()
    output = tmp_path / "report.pdf"

    pdf_report.save([], output)
    assert output.read_text() == "[]"

    written: list[tuple[Any, Path, str]] = []
    monkeypatch.setattr(
        pdf_report,
        "write_figure_image",
        lambda figure, path, format: written.append((figure, path, format)),
    )
    pdf_report.save([fig], output)
    assert written == [(fig, output, "pdf")]

    def fail_single(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(pdf_report, "write_figure_image", fail_single)
    with pytest.warns(RuntimeWarning, match="PDF export failed"):
        pdf_report.save([fig], output)
    assert output.read_text().startswith("[")

    monkeypatch.setattr(pdf_report, "PdfWriterType", None)
    with pytest.warns(RuntimeWarning, match="pypdf is required"):
        pdf_report.save([fig, fig], output)
    assert output.read_text().startswith("[")


def test_pdf_report_merges_and_async_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pa_core.viz import pdf_report

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    output = tmp_path / "merged.pdf"
    appended: list[bytes] = []

    class Writer:
        def append(self, buffer: io.BytesIO) -> None:
            appended.append(buffer.read())

        def write(self, handle: Any) -> None:
            handle.write(b"merged")

    monkeypatch.setattr(pdf_report, "PdfWriterType", Writer)
    monkeypatch.setattr(pdf_report, "figure_to_pdf_bytes", lambda figure: b"pdf")
    pdf_report.save([go.Figure(), go.Figure()], output)
    assert appended == [b"pdf", b"pdf"]
    assert output.read_bytes() == b"merged"

    cached: list[int] = []

    async def cache(figures: list[go.Figure], action: Any) -> None:
        cached.append(len(figures))
        action()

    monkeypatch.setattr(pdf_report, "run_with_browser_png_cache", cache)
    written: list[tuple[Any, Path, str]] = []
    monkeypatch.setattr(
        pdf_report,
        "write_figure_image",
        lambda figure, path, format: written.append((figure, path, format)),
    )
    asyncio.run(pdf_report.save_async([go.Figure()], output))
    assert cached == [1]
    assert len(written) == 1
    assert written[0][1:] == (output, "pdf")


def test_sweep_excel_metadata_image_and_async_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pa_core.reporting import sweep_excel

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    output = tmp_path / "sweep.xlsx"
    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    results = [{"combination_id": 4, "summary": summary}]
    monkeypatch.setattr(sweep_excel.risk_return, "make", lambda frame: go.Figure())
    png = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(png, format="PNG")
    monkeypatch.setattr(sweep_excel, "figure_to_png_bytes", lambda figure: png.getvalue())

    sweep_excel.export_sweep_results(results, str(output), metadata={"labels": ["a", "b"]})
    workbook = sweep_excel.openpyxl.load_workbook(output)
    assert {"Metadata", "Run4", "Summary", "SummaryStats"} <= set(workbook.sheetnames)
    assert workbook["Metadata"]["B2"].value == '["a", "b"]'
    assert len(workbook["Summary"]._images) == 1
    workbook.close()

    cached: list[int] = []

    async def cache(figures: list[go.Figure], action: Any) -> None:
        cached.append(len(figures))
        action()

    monkeypatch.setattr(sweep_excel, "run_with_browser_png_cache", cache)
    asyncio.run(sweep_excel.export_sweep_results_async(results, str(output)))
    assert cached == [1]
    workbook = sweep_excel.openpyxl.load_workbook(output)
    assert len(workbook["Summary"]._images) == 1
    workbook.close()
    assert sweep_excel._serialize_metadata_value(("a", 1)) == '["a", 1]'


def test_export_packet_render_figure_branches_and_async(monkeypatch: pytest.MonkeyPatch) -> None:
    from pa_core.reporting import export_packet

    sensitivity = pd.DataFrame(
        {
            "Parameter": ["mu_H"],
            "Base": [0.05],
            "Minus": [0.04],
            "Plus": [0.06],
            "Low": [-0.01],
            "High": [0.01],
            "DeltaAbs": [0.01],
        }
    )
    sensitivity.attrs["metric"] = "terminal_AnnReturn"
    sensitivity.attrs["units"] = "%"
    sensitivity.attrs["tickformat"] = ".2%"
    rendered = export_packet._packet_render_figures(
        figs=[], summary_df=pd.DataFrame(), inputs_dict={"_sensitivity_df": sensitivity}
    )
    assert len(rendered) >= 1

    calls: list[list[Any]] = []

    async def cache(figures: list[Any], action: Any) -> tuple[str, str]:
        calls.append(list(figures))
        return action()

    monkeypatch.setattr(export_packet, "run_with_browser_png_cache", cache)
    monkeypatch.setattr(export_packet, "create_export_packet", lambda **_kwargs: ("deck", "book"))
    result = asyncio.run(
        export_packet.create_export_packet_async(
            figs=[],
            summary_df=pd.DataFrame(),
            raw_returns_dict={},
            inputs_dict={"_sensitivity_df": sensitivity},
        )
    )
    assert result == ("deck", "book")
    assert len(calls) == 1
    assert len(calls[0]) == len(rendered) == 2
    assert [fig.to_plotly_json() for fig in calls[0]] == [fig.to_plotly_json() for fig in rendered]


def test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values() -> None:
    wizard = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    normalize = wizard["_normalize_config_chat_value"]
    unique = wizard["_stable_unique_text_items"]

    assert normalize({"items": ("one", "two")}) == {"items": ["one", "two"]}
    assert unique(None) == []
    assert unique(" one ") == ["one"]
    assert unique([" one ", "", "one", 2]) == ["one", "2"]
