from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataImportAgent:
    """Load asset time series from CSV or Excel and return long-form returns."""

    def __init__(self, *, date_col: str = "Date") -> None:
        self.date_col = date_col

    def load(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        else:
            raise ValueError("unsupported file type")
        if self.date_col not in df.columns:
            raise ValueError("date column missing")
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        long_df = df.melt(id_vars=[self.date_col], var_name="id", value_name="return")
        long_df = (
            long_df.dropna().sort_values([self.date_col, "id"]).reset_index(drop=True)
        )
        long_df.rename(columns={self.date_col: "date"}, inplace=True)
        return long_df
