from pathlib import Path

import openpyxl
import pandas as pd

from pa_core.reporting import export_to_excel


def test_shortfallprob_present(tmp_path: Path) -> None:
    inputs = {"Param": 1}
    summary = pd.DataFrame({"ShortfallProb": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]])}
    file_path = tmp_path / "out.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path))
    wb = openpyxl.load_workbook(file_path)
    ws = wb["Summary"]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    assert "ShortfallProb" in header
