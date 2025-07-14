from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

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

        if summary_frames:
            all_summary = pd.concat(summary_frames, ignore_index=True)
            all_summary.to_excel(writer, sheet_name="Summary", index=False)
