from __future__ import annotations

import io
from typing import Any, Dict, Iterable

import openpyxl
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

from ..viz import risk_return

__all__ = ["export_sweep_results"]


def export_sweep_results(
    results: Iterable[Dict[str, Any]], filename: str = "Sweep.xlsx"
) -> None:
    """Write sweep results to an Excel workbook with one sheet per combination."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary_frames = []
        for res in results:
            sheet = f"Run{res['combination_id']}"
            summary = res["summary"].copy()
            summary["ShortfallProb"] = summary.get("ShortfallProb", 0.0)
            summary.to_excel(writer, sheet_name=sheet, index=False)
            summary["Combination"] = sheet
            summary_frames.append(summary)
        all_summary: pd.DataFrame | None = None
        if summary_frames:
            all_summary = pd.concat(summary_frames, ignore_index=True)
            all_summary.to_excel(writer, sheet_name="Summary", index=False)

    wb = openpyxl.load_workbook(filename)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for column_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            col_idx = column_cells[0].column
            ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    if "Summary" in wb.sheetnames and all_summary is not None:
        ws = wb["Summary"]
        metrics = {"AnnReturn", "AnnVol", "VaR", "BreachProb", "TE"}
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        for idx, col_name in enumerate(header, 1):
            if col_name in metrics:
                col_letter = get_column_letter(idx)
                for cell in ws[col_letter][1:]:
                    cell.number_format = "0.00%"

        try:
            fig = risk_return.make(all_summary)
            img_bytes = fig.to_image(format="png")
            img = XLImage(io.BytesIO(img_bytes))
            ws.add_image(img, "H2")
        except Exception:
            pass

    wb.save(filename)
