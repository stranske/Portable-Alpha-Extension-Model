from __future__ import annotations

from typing import Mapping

import pandas as pd


def one_factor_deltas(
    base: pd.DataFrame, scenarios: Mapping[str, pd.DataFrame], *, value: str = "Sharpe"
) -> pd.Series:
    """Compute one-factor deltas for ``value`` relative to ``base``.

    Parameters
    ----------
    base: DataFrame
        Baseline results containing ``value`` column.
    scenarios: Mapping[str, DataFrame]
        Mapping of scenario name to results with only one parameter changed.
    value: str
        Column name to compute delta on. Defaults to ``"Sharpe"``.

    Returns
    -------
    Series
        Delta for each scenario sorted by absolute magnitude.
    """
    if value not in base:
        raise KeyError(f"{value} column missing from base DataFrame")

    base_val = float(base[value].mean())
    deltas: dict[str, float] = {}
    for name, df in scenarios.items():
        if value not in df:
            raise KeyError(f"{value} column missing from scenario '{name}'")
        deltas[name] = float(df[value].mean()) - base_val

    series = pd.Series(deltas)
    order = (
        pd.DataFrame(
            {
                "value": series,
                "abs": series.abs(),
                "name": series.index.astype(str),
            }
        )
        .sort_values(["abs", "name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return pd.Series(order["value"].to_numpy(), index=order["name"].to_list())
