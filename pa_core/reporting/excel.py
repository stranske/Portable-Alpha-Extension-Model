from __future__ import annotations

import io
from typing import Any, Dict, cast

import openpyxl
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

from ..viz import risk_return

__all__ = ["export_to_excel"]


def export_to_excel(
    inputs_dict: Dict[str, Any],
    summary_df: pd.DataFrame,
    raw_returns_dict: Dict[str, Any],
    filename: str = "Outputs.xlsx",
    *,
    pivot: bool = False,
) -> None:
    """Write inputs, summary, and raw returns into an Excel workbook.

    Parameters
    ----------
    inputs_dict : dict
        Mapping of input parameter names to values.
    summary_df : pandas.DataFrame
        Summary metrics to write to the ``Summary`` sheet.
    raw_returns_dict : dict[str, pandas.DataFrame]
        Per-agent returns matrices.
    filename : str, optional
        Destination Excel file name. Defaults to ``"Outputs.xlsx"``.
    pivot : bool, optional
        If ``True``, collapse all raw returns into a single ``AllReturns`` sheet
        in long format (``Sim``, ``Month``, ``Agent``, ``Return``). Otherwise a
        separate sheet is written per agent. Defaults to ``False``.
    """

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_inputs = pd.DataFrame(
            {
                "Parameter": list(inputs_dict.keys()),
                "Value": list(inputs_dict.values()),
            }
        )
        df_inputs.to_excel(writer, sheet_name="Inputs", index=False)
        summary_df = summary_df.copy()
        summary_df["ShortfallProb"] = summary_df.get("ShortfallProb", 0.0)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        if pivot:
            frames = []
            for name, df in raw_returns_dict.items():
                long_df = df.stack().rename("Return").reset_index()
                long_df.columns = ["Sim", "Month", "Return"]
                long_df["Agent"] = name
                frames.append(long_df[["Sim", "Month", "Agent", "Return"]])
            all_returns = pd.concat(frames, ignore_index=True)
            all_returns.to_excel(writer, sheet_name="AllReturns", index=False)
        else:
            for sheet_name, df in raw_returns_dict.items():
                safe_name = sheet_name if len(sheet_name) <= 31 else sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=True)

    wb = openpyxl.load_workbook(filename)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for column_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            col_idx = cast(int, column_cells[0].column)
            ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    if "Summary" in wb.sheetnames:
        ws = wb["Summary"]
        metrics = {"AnnReturn", "AnnVol", "VaR", "BreachProb", "TE"}
        header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        for idx, col_name in enumerate(header, 1):
            if col_name in metrics:
                col_letter = get_column_letter(idx)
                for cell in ws[col_letter][1:]:
                    cell.number_format = "0.00%"

        try:
            img_bytes = risk_return.make(summary_df).to_image(format="png")
            img = XLImage(io.BytesIO(img_bytes))
            ws.add_image(img, "H2")
        except Exception:
            pass

    wb.save(filename)
