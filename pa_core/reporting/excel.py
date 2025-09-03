from __future__ import annotations

import io
from typing import Any, Dict, cast

import openpyxl
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

from ..viz import risk_return, theme

__all__ = ["export_to_excel"]


def export_to_excel(
    inputs_dict: Dict[str, Any],
    summary_df: pd.DataFrame,
    raw_returns_dict: Dict[str, Any],
    filename: str = "Outputs.xlsx",
    *,
    pivot: bool = False,
    diff_config_df: pd.DataFrame | None = None,
    diff_metrics_df: pd.DataFrame | None = None,
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
        summary_df["ShortfallProb"] = summary_df.get(
            "ShortfallProb", theme.DEFAULT_SHORTFALL_PROB
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Optional: Sensitivity sheet if provided
        sens_maybe = inputs_dict.get("_sensitivity_df")
        sens_df: pd.DataFrame | None = (
            sens_maybe if isinstance(sens_maybe, pd.DataFrame) else None
        )
        if sens_df is not None and not sens_df.empty:
            # Write a concise view
            cols = [
                c
                for c in [
                    "Parameter",
                    "Base",
                    "Minus",
                    "Plus",
                    "Low",
                    "High",
                    "DeltaAbs",
                ]
                if c in sens_df.columns
            ]
            sens_df[cols].to_excel(writer, sheet_name="Sensitivity", index=False)
        # Optional diff sheets
        if diff_config_df is not None and not diff_config_df.empty:
            diff_config_df.to_excel(writer, sheet_name="ConfigDiff", index=False)
        if diff_metrics_df is not None and not diff_metrics_df.empty:
            diff_metrics_df.to_excel(writer, sheet_name="MetricDiff", index=False)

        # Write returns either pivoted or per-sheet
        if pivot:
            frames = []
            for name, df in raw_returns_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    stacked = df.stack()
                    stacked.name = "Return"
                    long_df = stacked.reset_index()
                    long_df.columns = ["Sim", "Month", "Return"]
                    long_df["Agent"] = name
                    frames.append(long_df[["Sim", "Month", "Agent", "Return"]])
            if frames:
                all_returns = pd.concat(frames, ignore_index=True)
                all_returns.to_excel(writer, sheet_name="AllReturns", index=False)
        else:
            for sheet_name, df in raw_returns_dict.items():
                if isinstance(df, pd.DataFrame):
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
        except (KeyError, ValueError, RuntimeError, OSError, MemoryError):
            # Some tests pass a minimal summary without expected columns like 'Agent' or 'AnnVol'; skip chart.
            pass

    # Best-effort: embed tornado image on Sensitivity sheet
    if "Sensitivity" in wb.sheetnames:
        try:
            from ..viz import tornado

            ws = wb["Sensitivity"]
            # Build figure from the written sheet
            values: Any = ws.values
            df = pd.DataFrame(values)
            df.columns = df.iloc[0]
            df = df.drop(index=0)
            # Convert the DataFrame to a Series mapping parameter names to values
            # Assumes the first column is the parameter name and the second column is the value
            # Adjust column names if needed
            param_col = df.columns[0]
            value_col = df.columns[1]
            series = df.set_index(param_col)[value_col].astype(float)
            fig = tornado.make(cast(pd.Series, series))
            img_bytes = fig.to_image(format="png")
            img = XLImage(io.BytesIO(img_bytes))
            ws.add_image(img, "H2")
        except Exception:
            # Non-fatal if renderer or data missing
            pass

    # Optional: write Attribution sheet if provided in inputs_dict
    attr_maybe = inputs_dict.get("_attribution_df")
    attr_df: pd.DataFrame | None = (
        attr_maybe if isinstance(attr_maybe, pd.DataFrame) else None
    )
    if attr_df is not None and not attr_df.empty:
        try:
            with pd.ExcelWriter(
                filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                cols = [c for c in ["Agent", "Sub", "Return"] if c in attr_df.columns]
                if cols:
                    attr_df[cols].to_excel(
                        writer, sheet_name="Attribution", index=False
                    )
        except Exception:
            pass

    # Best-effort: embed sunburst image on Attribution sheet
    if "Attribution" in wb.sheetnames:
        try:
            from ..viz import sunburst

            ws = wb["Attribution"]
            # Use attr_df directly instead of reconstructing from worksheet
            if attr_df is not None and not attr_df.empty:
                # Ensure required columns exist
                if {"Agent", "Sub", "Return"} <= set(attr_df.columns):
                    fig = sunburst.make(attr_df)
                    img_bytes = fig.to_image(format="png")
                    img = XLImage(io.BytesIO(img_bytes))
                    ws.add_image(img, "H2")
        except Exception:
            pass

    # Optional: write RiskAttribution sheet if provided
    risk_maybe = inputs_dict.get("_risk_attr_df")
    risk_df: pd.DataFrame | None = (
        risk_maybe if isinstance(risk_maybe, pd.DataFrame) else None
    )
    if risk_df is not None and not risk_df.empty:
        try:
            with pd.ExcelWriter(
                filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                cols = [
                    c
                    for c in [
                        "Agent",
                        "BetaVol",
                        "AlphaVol",
                        "CorrWithIndex",
                        "AnnVolApprox",
                        "TEApprox",
                    ]
                    if c in risk_df.columns
                ]
                if cols:
                    risk_df[cols].to_excel(
                        writer, sheet_name="RiskAttribution", index=False
                    )
        except Exception:
            pass

    wb.save(filename)

    # Optional: write Sleeve Trade-offs sheet if provided in inputs_dict
    trade_maybe = inputs_dict.get("_tradeoff_df")
    trade_df: pd.DataFrame | None = (
        trade_maybe if isinstance(trade_maybe, pd.DataFrame) else None
    )
    if trade_df is not None and not trade_df.empty:
        try:
            with pd.ExcelWriter(
                filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
            ) as writer:
                trade_df.to_excel(writer, sheet_name="SleeveTradeoffs", index=True)
        except Exception:
            pass
