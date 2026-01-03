from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from pa_core.reporting.sweep_excel import export_sweep_results

openpyxl = pytest.importorskip("openpyxl")
pptx = pytest.importorskip("pptx")

ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def test_export_sweep_results_writes_summary_and_run_sheet(tmp_path: Path, monkeypatch) -> None:
    from pa_core.reporting import sweep_excel

    results = [
        {
            "combination_id": 1,
            "summary": pd.DataFrame(
                {"Agent": ["Base"], "terminal_AnnReturn": [0.05], "monthly_AnnVol": [0.1]}
            ),
        }
    ]

    def _raise_make(*_args, **_kwargs):
        raise RuntimeError("no image")

    monkeypatch.setattr(sweep_excel.risk_return, "make", _raise_make)

    out_path = tmp_path / "sweep.xlsx"
    export_sweep_results(results, filename=str(out_path))

    wb = openpyxl.load_workbook(out_path)
    assert {"Run1", "Summary"} <= set(wb.sheetnames)

    for name in ["Run1", "Summary"]:
        assert wb[name].freeze_panes == "A2"

    header = [cell.value for cell in next(wb["Summary"].iter_rows(max_row=1))]
    assert "terminal_ShortfallProb" in header


def test_export_sweep_results_writes_summary_when_empty(tmp_path: Path) -> None:
    out_path = tmp_path / "sweep_empty.xlsx"
    export_sweep_results([], filename=str(out_path))

    wb = openpyxl.load_workbook(out_path)
    assert "Summary" in wb.sheetnames

    header = [cell.value for cell in next(wb["Summary"].iter_rows(max_row=1))]
    assert "terminal_ShortfallProb" in header


def test_create_export_packet_writes_files_and_manifest_slide(tmp_path: Path) -> None:
    from pa_core.reporting.export_packet import create_export_packet

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    raw_returns = {"Base": pd.DataFrame([[0.01, 0.02]], columns=[0, 1])}
    inputs = {"foo": 1}

    class _Fig:
        def to_image(self, format: str = "png", engine: str = "kaleido") -> bytes:
            return ONE_PX_PNG

    manifest = {
        "git_commit": "abc123",
        "timestamp": "2024-01-01T00:00:00Z",
        "seed": 42,
        "data_files": {"data.csv": "a" * 64},
        "cli_args": {"mode": "returns"},
        "config": {"N_SIMULATIONS": 1000, "N_MONTHS": 12},
    }

    pptx_path, excel_path = create_export_packet(
        figs=[_Fig()],
        summary_df=summary,
        raw_returns_dict=raw_returns,
        inputs_dict=inputs,
        base_filename=tmp_path / "packet",
        alt_texts=["Summary chart"],
        manifest=manifest,
    )

    assert Path(pptx_path).exists()
    assert Path(excel_path).exists()

    presentation = pptx.Presentation(pptx_path)
    assert len(presentation.slides) == 5


def test_create_export_packet_adds_tornado_slide(tmp_path: Path) -> None:
    from pa_core.reporting.export_packet import create_export_packet

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    raw_returns = {"Base": pd.DataFrame([[0.01, 0.02]], columns=[0, 1])}
    sens_df = pd.DataFrame(
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
    sens_df.attrs.update({"metric": "terminal_AnnReturn", "units": "%", "tickformat": ".2%"})
    inputs = {"_sensitivity_df": sens_df}

    class _Fig:
        layout = type("Layout", (), {"title": type("Title", (), {"text": "Summary"})()})()

    pptx_path, _excel_path = create_export_packet(
        figs=[_Fig()],
        summary_df=summary,
        raw_returns_dict=raw_returns,
        inputs_dict=inputs,
        base_filename=tmp_path / "packet",
    )

    presentation = pptx.Presentation(pptx_path)
    assert len(presentation.slides) == 5


def test_export_to_excel_includes_tornado_snapshot(tmp_path: Path) -> None:
    from pa_core.reporting.excel import export_to_excel

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    sens_df = pd.DataFrame(
        {
            "Parameter": ["mu_H", "sigma_H"],
            "Base": [0.05, 0.05],
            "Minus": [0.04, 0.045],
            "Plus": [0.055, 0.052],
            "Low": [-0.01, -0.005],
            "High": [0.005, 0.002],
            "DeltaAbs": [0.01, 0.005],
        }
    )
    sens_df.attrs.update({"metric": "terminal_AnnReturn", "units": "%", "tickformat": ".2%"})
    inputs = {"_sensitivity_df": sens_df}
    out_path = tmp_path / "outputs.xlsx"

    export_to_excel(inputs, summary, {}, filename=str(out_path))

    wb = openpyxl.load_workbook(out_path)
    assert "Sensitivity" in wb.sheetnames
    ws = wb["Sensitivity"]
    assert ws._images, "Expected tornado chart snapshot in Sensitivity sheet"


def test_export_to_excel_includes_sunburst_snapshot(tmp_path: Path) -> None:
    from pa_core.reporting.excel import export_to_excel

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    attr_df = pd.DataFrame({"Agent": ["Base"], "Sub": ["Core"], "Return": [0.01]})
    risk_df = pd.DataFrame(
        {
            "Agent": ["Base"],
            "BetaVol": [0.1],
            "AlphaVol": [0.05],
            "CorrWithIndex": [0.8],
            "AnnVolApprox": [0.12],
            "TEApprox": [0.03],
        }
    )
    inputs = {"_attribution_df": attr_df, "_risk_attr_df": risk_df}
    out_path = tmp_path / "outputs.xlsx"

    export_to_excel(inputs, summary, {}, filename=str(out_path))

    wb = openpyxl.load_workbook(out_path)
    assert {"Attribution", "RiskAttribution"} <= set(wb.sheetnames)
    ws = wb["Attribution"]
    assert ws._images, "Expected sunburst chart snapshot in Attribution sheet"


def test_export_to_excel_includes_sleeve_attribution_sheet(tmp_path: Path) -> None:
    from pa_core.reporting.excel import export_to_excel

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    sleeve_attr = pd.DataFrame(
        {
            "Agent": ["ExternalPA", "ActiveExt", "Total"],
            "ReturnContribution": [0.12, 0.08, 0.2],
        }
    )
    inputs = {"_sleeve_attribution_df": sleeve_attr}
    out_path = tmp_path / "outputs.xlsx"

    export_to_excel(inputs, summary, {}, filename=str(out_path))

    wb = openpyxl.load_workbook(out_path)
    assert "sleeve_attribution" in wb.sheetnames
