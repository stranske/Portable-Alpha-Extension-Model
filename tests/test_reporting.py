import pandas as pd
import openpyxl
from pathlib import Path
from pa_core.reporting import export_to_excel


def test_export_to_excel_sheets(tmp_path: Path):
    inputs = {"a": 1, "b": 2}
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "out.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path))
    wb = openpyxl.load_workbook(file_path)
    assert set(wb.sheetnames) == {"Inputs", "Summary", "Base"}
