"""App-specific adapter for PAEM.

The only app-specific piece the shared ``baseline_kit`` needs: load a config,
apply an input patch, run the real Monte Carlo simulation deterministically, and
reduce the per-agent summary to flat, comparison-friendly metrics. Everything
else (directional checks, invariants, golden masters, coverage manifest) is
generic and lives in ``baseline_kit``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "examples" / "scenarios" / "my_first_scenario.yml"
SEED = 42

# Small but stable run for deterministic baselines (the demo scenario funds all
# three sleeves: external_pa=100, active_ext=50, internal_pa=150 / total=300).
BASE_PATCH: dict[str, Any] = {"N_SIMULATIONS": 400, "N_MONTHS": 12}

# Deterministic 12-month benchmark index (no external data file needed).
INDEX = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.01, 0.005, -0.015, 0.02, 0.01, -0.005])

# Numeric metric columns from the simulation summary table.
METRIC_COLS = [
    "terminal_AnnReturn",
    "terminal_ExcessReturn",
    "monthly_AnnVol",
    "monthly_VaR",
    "monthly_CVaR",
    "terminal_CVaR",
    "monthly_MaxDD",
    "monthly_TimeUnderWater",
    "monthly_BreachProb",
    "terminal_ShortfallProb",
    "monthly_TE",
]
_PROB_COLS = ("monthly_BreachProb", "terminal_ShortfallProb", "monthly_TimeUnderWater")


@functools.lru_cache(maxsize=1)
def _base_config():
    from pa_core.config import load_config

    return load_config(str(SCENARIO))


def base_field(name: str) -> float:
    """The loaded (post-conversion) value of a config field, for relative patches."""
    return float(getattr(_base_config(), name))


def run(patch: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run the simulation with an input patch; return the summary indexed by Agent."""
    from pa_core.facade import RunOptions, run_single

    data = _base_config().model_dump()
    data.update({**BASE_PATCH, **(patch or {})})
    cfg = _base_config().__class__.model_validate(data)
    artifacts = run_single(cfg, INDEX, RunOptions(seed=SEED))
    return artifacts.summary.set_index("Agent")


def agent_names(summary: pd.DataFrame) -> list[str]:
    return [str(a) for a in summary.index]


def metric(summary: pd.DataFrame, agent_substr: str, col: str) -> float:
    """A single metric for the first agent whose name contains ``agent_substr``."""
    rows = [a for a in summary.index if agent_substr.lower() in str(a).lower()]
    if not rows:
        return float("nan")
    value = summary.loc[rows[0], col]
    return float(value) if pd.notna(value) else float("nan")


def flat_metrics(summary: pd.DataFrame) -> dict[str, float]:
    """Every (agent, numeric-metric) cell flattened to ``agent.col`` -> value.

    NaNs (e.g. tracking error for the benchmark agent) are dropped so the golden
    master is stable.
    """
    out: dict[str, float] = {}
    for agent in summary.index:
        for col in METRIC_COLS:
            if col in summary.columns:
                value = summary.loc[agent, col]
                if pd.notna(value):
                    out[f"{agent}.{col}"] = float(value)
    return out


def is_probability_col(col: str) -> bool:
    return col in _PROB_COLS
