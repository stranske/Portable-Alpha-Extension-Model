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
        monthly_rule: Literal["ME", "MS"] = "ME",
        min_obs: int = 36,
        # Optional I/O parsing controls
        sheet_name: str | int | None = None,
        na_values: list[str] | None = None,
        decimal: str = ".",
        thousands: str | None = None,
    ) -> None:
        self.date_col = date_col
        self.id_col = id_col
        self.value_col = value_col
        self.wide = wide
        self.value_type = value_type
        self.frequency = frequency
        if monthly_rule not in {"ME", "MS"}:
            raise ValueError("monthly_rule must be 'ME' or 'MS'")
        self.monthly_rule = monthly_rule
        self.min_obs = min_obs
        # I/O parsing options (safe defaults keep existing behavior)
        self.sheet_name = sheet_name
        self.na_values = na_values
        self.decimal = decimal
        self.thousands = thousands
        self.metadata: Dict[str, Any] | None = None

    def load(self, path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            csv_kwargs: Dict[str, Any] = {}
            if self.na_values is not None:
                csv_kwargs["na_values"] = self.na_values
            if self.decimal:
                csv_kwargs["decimal"] = self.decimal
            if self.thousands is not None:
                csv_kwargs["thousands"] = self.thousands
            df = pd.read_csv(p, **csv_kwargs)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            if self.sheet_name is None:
                df = pd.read_excel(p)
            else:
                df = pd.read_excel(p, sheet_name=self.sheet_name)
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
            long_df = cast(
                pd.DataFrame, df[[self.date_col, self.id_col, self.value_col]].copy()
            )

        # Coerce value column to numeric early; treat non-numeric as NA and drop
        long_df[self.value_col] = pd.to_numeric(
            long_df[self.value_col], errors="coerce"
        )
        long_df = cast(pd.DataFrame, long_df.dropna(subset=[self.value_col]))
        long_df = long_df.sort_values(by=[self.date_col, self.id_col])
        long_df = long_df.rename(
            columns={self.date_col: "date", self.id_col: "id", self.value_col: "value"}
        )

        if self.value_type == "prices":
            long_df = long_df.sort_values(by=["id", "date"])
            if self.frequency == "daily":
                wide = long_df.pivot(index="date", columns="id", values="value")
                month_start = wide.resample("MS").first()
                month_end = wide.resample("ME").last()
                if self.monthly_rule == "MS":
                    month_start_period = month_start.copy()
                    month_end_period = month_end.copy()
                    month_start_period.index = month_start_period.index.to_period("M")
                    month_end_period.index = month_end_period.index.to_period("M")
                    returns = (month_end_period / month_start_period) - 1
                    returns = returns.dropna(how="all")
                    returns.index = month_start.index[: len(returns)]
                else:
                    first = (month_start.shift(-1) / month_start - 1).iloc[:1]
                    rets = month_end.pct_change().iloc[1:]
                    returns = pd.concat([first, rets])
                    returns.index = month_end.index[: len(returns)]
                returns = returns.melt(
                    ignore_index=False, var_name="id", value_name="return"
                )
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
                # Compound daily returns into month-end returns
                long_df = (
                    long_df.set_index("date")
                    .groupby("id")["return"]
                    .apply(
                        lambda s: (1 + s).resample(self.monthly_rule).prod() - 1
                    )
                    .dropna()
                    .reset_index()
                )

        # Ensure we have data after parsing
        if long_df.empty:
            raise ValueError("no valid data after parsing")

        diffs = long_df.groupby("id")["date"].diff()
        if (diffs.dropna() <= pd.Timedelta(0)).any():
            raise ValueError("dates must be strictly increasing within each id")

        counts = cast(pd.Series, long_df.groupby("id").size())
        bad = cast(pd.Series, counts[counts < self.min_obs])
        if not bad.empty:
            max_ids = 10
            id_list = [str(x) for x in sorted(bad.index.tolist())]
            shown_ids = id_list[:max_ids]
            ids_str = ", ".join(shown_ids)
            if len(id_list) > max_ids:
                ids_str += f", ... and {len(id_list) - max_ids} more"
            raise ValueError(f"insufficient data for ids: {ids_str}")

        self.metadata = {
            "source_file": str(p),
            "value_type": self.value_type,
            "frequency": self.frequency,
            "monthly_rule": self.monthly_rule,
            "wide": self.wide,
            "io": {
                "sheet_name": self.sheet_name,
                "na_values": (
                    list(self.na_values) if self.na_values is not None else None
                ),
                "decimal": self.decimal,
                "thousands": self.thousands,
            },
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
            "monthly_rule": self.monthly_rule,
            "min_obs": self.min_obs,
            # I/O options
            "sheet_name": self.sheet_name,
            "na_values": list(self.na_values) if self.na_values is not None else None,
            "decimal": self.decimal,
            "thousands": self.thousands,
        }
        Path(path).write_text(yaml.safe_dump(data))

    @classmethod
    def from_template(cls, path: str | Path) -> "DataImportAgent":
        """Create an instance from a previously saved mapping template."""
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError(
                f"Invalid template file: expected YAML dictionary but got {type(data).__name__}"
            )
        # Only keep known keys with expected types
        allowed = {
            "date_col": str,
            "id_col": str,
            "value_col": str,
            "wide": bool,
            "value_type": str,
            "frequency": str,
            "monthly_rule": str,
            "min_obs": int,
            # I/O options
            "sheet_name": (str | int | type(None)),
            "na_values": (list | type(None)),
            "decimal": str,
            "thousands": (str | type(None)),
        }
        kwargs: Dict[str, Any] = {}
        for key, typ in allowed.items():
            if key in data:
                value = data[key]
                # Handle typing for unions (e.g., str | int | type(None))
                if isinstance(typ, tuple):
                    expected_types = typ
                else:
                    expected_types = (typ,)
                if not isinstance(value, expected_types):
                    raise TypeError(
                        f"Template key '{key}' expects type {expected_types} but got {type(value).__name__}"
                    )
                kwargs[key] = value
        return cls(**kwargs)
