from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pa_core.reporting import export_to_excel

openpyxl: Any = pytest.importorskip("openpyxl")


def test_export_to_excel_sheets(tmp_path: Path):
    inputs = {"a": 1, "b": 2}
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "out.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path))
    wb = openpyxl.load_workbook(file_path)
    assert set(wb.sheetnames) == {"Inputs", "Summary", "Base"}


def test_export_to_excel_pivot(tmp_path: Path):
    inputs = {"x": 1}
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {
        "Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1]),
        "Ext": pd.DataFrame([[0.3, 0.4]], columns=[0, 1]),
    }
    file_path = tmp_path / "pivot.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path), pivot=True)
    wb = openpyxl.load_workbook(file_path)
    assert set(wb.sheetnames) == {"Inputs", "Summary", "AllReturns"}
    ws = wb["AllReturns"]
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    assert header == ["Sim", "Month", "Agent", "Return"]


def test_export_to_excel_adds_attribution_and_risk_sheets(tmp_path: Path) -> None:
    inputs = {
        "_attribution_df": pd.DataFrame(
            {"Agent": ["Base"], "Sub": ["Core"], "Return": [0.01]}
        ),
        "_risk_attr_df": pd.DataFrame(
            {
                "Agent": ["Base"],
                "BetaVol": [0.1],
                "AlphaVol": [0.05],
                "CorrWithIndex": [0.8],
                "AnnVolApprox": [0.12],
                "TEApprox": [0.03],
            }
        ),
    }
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "with_attr.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    wb = openpyxl.load_workbook(file_path)
    assert {"Attribution", "RiskAttribution"} <= set(wb.sheetnames)
