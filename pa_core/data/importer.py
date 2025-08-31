from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Set, cast

import pandas as pd
import yaml


class DataImportAgent:
    """Load asset time series from CSV or Excel and return long-form returns.

    Supports both wide and already-long formats and can transform price
    series to returns. If ``frequency`` is ``daily`` the returns are
    compounded to monthly. After :meth:`load` the agent exposes a
    ``metadata`` attribute with the applied mappings for UI consumption.
    """

    def __init__(
        self,
        *,
        date_col: str = "Date",
        id_col: str = "Id",
        value_col: str = "Return",
        wide: bool = True,
        value_type: Literal["returns", "prices"] = "returns",
        frequency: Literal["monthly", "daily"] = "monthly",
        min_obs: int = 36,
    ) -> None:
        self.date_col = date_col
        self.id_col = id_col
        self.value_col = value_col
        self.wide = wide
        self.value_type = value_type
        self.frequency = frequency
        self.min_obs = min_obs
        self.metadata: Dict[str, Any] | None = None

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

        long_df: pd.DataFrame
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
            long_df = cast(pd.DataFrame, df[[self.date_col, self.id_col, self.value_col]].copy())

        long_df = cast(pd.DataFrame, long_df.dropna(subset=[self.value_col]))
        long_df = long_df.sort_values(by=[self.date_col, self.id_col])
        long_df = long_df.rename(
            columns={self.date_col: "date", self.id_col: "id", self.value_col: "value"}
        )

        if self.value_type == "prices":
            long_df = long_df.sort_values(by=["id", "date"])
            if self.frequency == "daily":
                wide = long_df.pivot(index="date", columns="id", values="value")
                month_end = wide.resample("ME").last()
                month_start = wide.resample("MS").first()
                first = (month_start.shift(-1) / month_start - 1).iloc[:1]
                rets = month_end.pct_change().iloc[1:]
                returns = pd.concat([first, rets])
                returns.index = month_end.index[: len(returns)]
                returns = returns.melt(ignore_index=False, var_name="id", value_name="return")
                long_df = cast(
                    pd.DataFrame,
                    (
                        returns.dropna()
                        .reset_index()
                        .rename(columns={"index": "date"})[["id", "date", "return"]]
                    ),
                )
            else:
                long_df["return"] = long_df.groupby("id")["value"].pct_change()
                long_df = long_df.dropna(subset=["return"])
                long_df = long_df.drop(columns=["value"])
        else:
            long_df = long_df.rename(columns={"value": "return"})
            long_df = long_df.dropna(subset=["return"])

            if self.frequency == "daily":
                long_df = (
                    long_df.set_index("date")
                    .groupby("id")["return"]
                    .apply(lambda s: (1 + s * 365).resample("ME").prod() - 1)
                    .dropna()
                    .reset_index()
                )

        diffs = long_df.groupby("id")["date"].diff()
        if (diffs.dropna() <= pd.Timedelta(0)).any():
            raise ValueError("dates must be strictly increasing within each id")

        counts = cast(pd.Series, long_df.groupby("id").size())
        bad = cast(pd.Series, counts[counts < self.min_obs])
        if not bad.empty:
            max_ids = 10
            id_list = sorted(bad.index.astype(str))
            shown_ids = id_list[:max_ids]
            ids_str = ", ".join(shown_ids)
            if len(id_list) > max_ids:
                ids_str += f", ... and {len(id_list) - max_ids} more"
            raise ValueError(f"insufficient data for ids: {ids_str}")

        self.metadata = {
            "source_file": str(p),
            "value_type": self.value_type,
            "frequency": self.frequency,
            "wide": self.wide,
            "columns": {
                "date": self.date_col,
                "id": self.id_col,
                "value": self.value_col,
            },
        }

        return cast(pd.DataFrame, long_df.reset_index(drop=True))

    # ------------------------------------------------------------------
    # Mapping templates

    def save_template(self, path: str | Path) -> None:
        """Persist the current column mapping and options to a YAML file.

        The template captures the initializer arguments so that a new
        :class:`DataImportAgent` can be reconstructed later or in a UI
        workflow. Only simple built-in types are stored to keep templates
        portable.
        """

        data = {
            "date_col": self.date_col,
            "id_col": self.id_col,
            "value_col": self.value_col,
            "wide": self.wide,
            "value_type": self.value_type,
            "frequency": self.frequency,
            "min_obs": self.min_obs,
        }
        Path(path).write_text(yaml.safe_dump(data))

    @classmethod
    def from_template(cls, path: str | Path) -> "DataImportAgent":
        """Create an instance from a previously saved mapping template."""
        import yaml
        
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError(f"Invalid template file: expected YAML dictionary but got {type(data).__name__}")

