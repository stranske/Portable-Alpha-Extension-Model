from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

__all__ = [
    "one_factor_deltas",
]


def one_factor_deltas(
    *,
    params: dict[str, float],
    steps: dict[str, float],
    evaluator: Callable[[dict[str, float]], float],
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute one-factor +/- step deltas for an arbitrary evaluator.

    Contract:
    - inputs: params dict of baseline values; steps dict mapping key->step size
    - evaluator: function(params)-> float metric (higher-is-better)
    - returns: DataFrame with columns [Parameter, Base, Minus, Plus, Low, High, DeltaAbs]

    Notes:
    - If a key is missing from params or steps, it is skipped.
    - Evaluator exceptions for a key are swallowed and that key is skipped.
    """

    base = evaluator(dict(params))
    records: list[tuple[str, float, float, float, float, float, float]] = []
    keys_iter: Iterable[str] = keys if keys is not None else steps.keys()

    for k in keys_iter:
        if k not in params or k not in steps:
            continue
        step = steps[k]
        # minus
        p_minus = dict(params)
        p_minus[k] = p_minus[k] - step
        # plus
        p_plus = dict(params)
        p_plus[k] = p_plus[k] + step
        try:
            m_minus = evaluator(p_minus)
            m_plus = evaluator(p_plus)
        except Exception:
            # Skip keys that cause evaluator failure
            continue
        low = m_minus - base
        high = m_plus - base
        delta_abs = max(abs(low), abs(high))
        records.append((k, base, m_minus, m_plus, low, high, delta_abs))

    df = pd.DataFrame.from_records(
        records,
        columns=["Parameter", "Base", "Minus", "Plus", "Low", "High", "DeltaAbs"],
    )
    df.sort_values(
        ["DeltaAbs", "Parameter"],
        ascending=[False, True],
        inplace=True,
        kind="mergesort",
    )
    df.reset_index(drop=True, inplace=True)
    return df
