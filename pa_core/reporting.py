from __future__ import annotations
import pandas as pd

__all__ = ["export_to_excel"]


def export_to_excel(inputs_dict, summary_df, raw_returns_dict, filename="Outputs.xlsx"):
    """Write inputs, summary, and raw returns into an Excel workbook."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_inputs = pd.DataFrame({
            "Parameter": list(inputs_dict.keys()),
            "Value": list(inputs_dict.values()),
        })
        df_inputs.to_excel(writer, sheet_name="Inputs", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, df in raw_returns_dict.items():
            safe_name = sheet_name if len(sheet_name) <= 31 else sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=True)
    print(f"Exported results to {filename}")
