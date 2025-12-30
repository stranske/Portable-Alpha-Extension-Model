from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pandas as pd
from rich.console import Console
from rich.table import Table

__all__ = ["print_summary", "print_run_diff"]


def print_summary(summary: pd.DataFrame | Mapping[str, float]) -> None:
    """Pretty-print summary metrics using Rich.

    Parameters
    ----------
    summary : pandas.DataFrame or mapping
        Summary metrics with columns or key names representing metrics.
    """
    console = Console()
    if isinstance(summary, pd.DataFrame):
        # Convert DataFrame to dict of columns -> list values
        df = cast(pd.DataFrame, summary)
        data: dict[str, list[object]] = {str(col): df[col].tolist() for col in df.columns}
    else:
        # Convert mapping to single-row dataframe-like dict
        data = {str(k): [v] for k, v in dict(summary).items()}

    table = Table(show_header=True, header_style="bold magenta")
    columns: list[str] = list(data.keys())
    for col in columns:
        table.add_column(col)
    n_rows = len(next(iter(data.values()))) if data else 0
    for i in range(n_rows):
        row = [
            (f"{data[c][i]:.4f}" if isinstance(data[c][i], (float, int)) else str(data[c][i]))
            for c in columns
        ]
        table.add_row(*row)
    console.print(table)


def _format_cell(value: object, col_name: str) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if col_name == "Delta":
            return f"{value:+.4f}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _build_diff_table(df: pd.DataFrame, *, title: str, max_rows: int) -> Table:
    view = df.copy()
    if "Delta" in view.columns:
        delta = pd.to_numeric(view["Delta"], errors="coerce")
        if delta.notna().any():
            view = (
                view.assign(_delta_abs=delta.abs())
                .sort_values("_delta_abs", ascending=False)
                .drop(columns="_delta_abs")
            )
    if len(view) > max_rows:
        view = view.iloc[:max_rows]
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in view.columns:
        table.add_column(str(col))
    for row in view.itertuples(index=False):
        table.add_row(*[_format_cell(value, col) for col, value in zip(view.columns, row)])
    return table


def print_run_diff(
    cfg_diff_df: pd.DataFrame | None,
    metric_diff_df: pd.DataFrame | None,
    *,
    max_rows: int = 8,
) -> None:
    """Print concise config/metric diffs comparing to the previous run."""
    console = Console()
    printed = False
    if cfg_diff_df is not None and not cfg_diff_df.empty:
        console.print(
            _build_diff_table(cfg_diff_df, title="Config Changes vs Previous", max_rows=max_rows)
        )
        printed = True
    if metric_diff_df is not None and not metric_diff_df.empty:
        console.print(
            _build_diff_table(metric_diff_df, title="Metric Changes vs Previous", max_rows=max_rows)
        )
        printed = True
    if not printed:
        console.print("No changes detected vs previous run.")
