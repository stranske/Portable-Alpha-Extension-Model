from __future__ import annotations
import pandas as pd

__all__ = ["export_to_excel"]


def export_to_excel(
    inputs_dict,
    summary_df,
    raw_returns_dict,
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
        df_inputs = pd.DataFrame({
            "Parameter": list(inputs_dict.keys()),
            "Value": list(inputs_dict.values()),
        })
        df_inputs.to_excel(writer, sheet_name="Inputs", index=False)
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

    print(f"Exported results to {filename}")
