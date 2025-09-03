from __future__ import annotations

from typing import Mapping

import pandas as pd
from rich.console import Console
from rich.table import Table

__all__ = ["print_summary"]


def print_summary(summary: pd.DataFrame | Mapping[str, float]) -> None:
    """Pretty-print summary metrics using Rich.

    Parameters
    ----------
    summary : pandas.DataFrame or mapping
        Summary metrics with columns or key names representing metrics.
    """
    console = Console()
    if isinstance(summary, pd.DataFrame):
        data: dict[str, list[object]] = {
            str(k): list(v) for k, v in summary.items()
        }  # type: ignore[assignment]
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
            (
                f"{data[c][i]:.4f}"
                if isinstance(data[c][i], (float, int))
                else str(data[c][i])
            )
            for c in columns
        ]
        table.add_row(*row)
    console.print(table)
