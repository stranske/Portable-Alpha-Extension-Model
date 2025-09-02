from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .config import ModelConfig
from .orchestrator import SimulatorOrchestrator


def suggest_sleeve_sizes(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    step: float = 0.25,
    seed: int | None = None,
    max_evals: int | None = 500,
) -> pd.DataFrame:
    """Suggest sleeve allocations that respect risk constraints.

    Performs a simple grid search over capital allocations for the three
    sleeves (external portable alpha, active extension and internal PA).
    Combinations where any sleeve breaches the supplied risk limits are
    discarded.

    Parameters
    ----------
    cfg:
        Base :class:`~pa_core.config.ModelConfig` used as a template.
    idx_series:
        Benchmark return series used by :class:`SimulatorOrchestrator`.
    max_te:
        Maximum allowed tracking error per sleeve.
    max_breach:
        Maximum allowed breach probability per sleeve.
    max_cvar:
        Absolute CVaR cap per sleeve.
    step:
        Grid step as a fraction of ``total_fund_capital``.
    seed:
        Optional random seed for reproducibility.
    max_evals:
        If set and the Cartesian grid would exceed this number of
        combinations, a random subset of at most ``max_evals`` points is
        evaluated. This prevents exponential runtime as ``step`` becomes
        small.

    Returns
    -------
    pandas.DataFrame
        Table of feasible capital combinations and associated metrics.
    """

    total = cfg.total_fund_capital
    grid = np.arange(0.0, total + 1e-9, total * step)

    combos = [
        (ext_cap, act_cap)
        for ext_cap, act_cap in itertools.product(grid, repeat=2)
        if (total - ext_cap - act_cap) >= 0
    ]
    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_evals, replace=False)
        combos = [combos[i] for i in idx]

    records: list[dict[str, float]] = []
    for ext_cap, act_cap in combos:
        int_cap = total - ext_cap - act_cap
        test_cfg = cfg.model_copy(
            update={
                "external_pa_capital": float(ext_cap),
                "active_ext_capital": float(act_cap),
                "internal_pa_capital": float(int_cap),
            }
        )
        orch = SimulatorOrchestrator(test_cfg, idx_series)
        _, summary = orch.run(seed=seed)
        meets = True
        metrics: dict[str, float] = {}
        for agent in ["ExternalPA", "ActiveExt", "InternalPA"]:
            sub = summary[summary["Agent"] == agent]
            if sub.empty:
                continue
            row = sub.iloc[0]
            te = row["TE"] if row["TE"] is not None else 0.0
            bprob = row["BreachProb"]
            cvar = row["CVaR"]
            metrics[f"{agent}_TE"] = te
            metrics[f"{agent}_BreachProb"] = bprob
            metrics[f"{agent}_CVaR"] = cvar
            if te > max_te or bprob > max_breach or abs(cvar) > max_cvar:
                meets = False
        if meets:
            record = {
                "external_pa_capital": float(ext_cap),
                "active_ext_capital": float(act_cap),
                "internal_pa_capital": float(int_cap),
            }
            record.update(metrics)
            records.append(record)
    return pd.DataFrame.from_records(records)
