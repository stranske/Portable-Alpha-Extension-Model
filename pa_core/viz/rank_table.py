from __future__ import annotations

import pandas as pd


def make(df_summary: pd.DataFrame, metric: str = "terminal_AnnReturn") -> pd.DataFrame:
    """Return table sorted by the chosen metric."""
    return df_summary.sort_values(metric, ascending=False).reset_index(drop=True)
