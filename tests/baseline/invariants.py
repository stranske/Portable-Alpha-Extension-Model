"""PAEM economic/structural invariants -- must hold for every agent, every run.

Bounds are app-specific; the result type and assertion helper are shared
(``baseline_kit.InvariantResult`` / ``assert_invariants``).
"""

from __future__ import annotations

import math

import pandas as pd
from baseline_kit import InvariantResult

from . import adapter


def check_run(summary: pd.DataFrame) -> list[InvariantResult]:
    results: list[InvariantResult] = []

    def add(name, ok, detail, severity="error"):
        results.append(InvariantResult(name, bool(ok), severity, detail))

    for agent in summary.index:
        row = summary.loc[agent]

        # Probabilities must lie in [0, 1].
        for col in adapter.METRIC_COLS:
            if adapter.is_probability_col(col) and col in row and pd.notna(row[col]):
                add(f"{agent}.{col}_in_unit_interval", 0.0 <= float(row[col]) <= 1.0, f"{col}={row[col]}")

        # Volatility non-negative and finite.
        vol = float(row["monthly_AnnVol"])
        add(f"{agent}.vol_non_negative", math.isfinite(vol) and vol >= 0, f"monthly_AnnVol={vol}")

        # Annualized return can't be worse than total loss (> -1).
        ret = float(row["terminal_AnnReturn"])
        add(f"{agent}.terminal_return_gt_neg1", math.isfinite(ret) and ret > -1.0, f"terminal_AnnReturn={ret}")

        # Tracking error, when present, is non-negative.
        if "monthly_TE" in row and pd.notna(row["monthly_TE"]):
            te = float(row["monthly_TE"])
            add(f"{agent}.TE_non_negative", te >= 0, f"monthly_TE={te}")

        # Max drawdown is a loss fraction in [-1, 0].
        if "monthly_MaxDD" in row and pd.notna(row["monthly_MaxDD"]):
            dd = float(row["monthly_MaxDD"])
            add(f"{agent}.maxdd_in_range", -1.0 - 1e-9 <= dd <= 1e-9, f"monthly_MaxDD={dd}")

    return results
