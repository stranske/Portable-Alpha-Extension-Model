from __future__ import annotations

from pathlib import Path
from typing import Literal, Set

import pandas as pd


class DataImportAgent:
    """Load asset time series from CSV or Excel and return long-form returns.

    Supports both wide and already-long formats and can transform price
    series to returns. Optionally aggregates higher frequency returns to
    monthly via compounding.
    """

    def __init__(
        self,
        *,
        date_col: str = "Date",
        id_col: str = "Id",
        value_col: str = "Return",
        wide: bool = True,
        value_type: Literal["returns", "prices"] = "returns",
        to_monthly: bool = False,
    ) -> None:
        self.date_col = date_col
        self.id_col = id_col
        self.value_col = value_col
        self.wide = wide
        self.value_type = value_type
        self.to_monthly = to_monthly

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

        if self.wide:
            long_df = df.melt(
                id_vars=[self.date_col],
                var_name=self.id_col,
                value_name=self.value_col,
            )
        else:
            required: Set[str] = {self.id_col, self.value_col}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            long_df = df[[self.date_col, self.id_col, self.value_col]].copy()

        long_df.dropna(subset=[self.value_col], inplace=True)
        long_df.sort_values([self.date_col, self.id_col], inplace=True)
        long_df.rename(
            columns={self.date_col: "date", self.id_col: "id", self.value_col: "value"},
            inplace=True,
        )

        if self.value_type == "prices":
            long_df = long_df.sort_values(["id", "date"])
            long_df["value"] = long_df.groupby("id")["value"].pct_change()
            long_df.dropna(subset=["value"], inplace=True)

        long_df.rename(columns={"value": "return"}, inplace=True)

        if self.to_monthly:
            long_df = (
                long_df.set_index("date")
                .groupby("id")["return"]
                .apply(lambda s: (1 + s).resample("M").prod() - 1)
                .reset_index()
            )

        return long_df.reset_index(drop=True)
