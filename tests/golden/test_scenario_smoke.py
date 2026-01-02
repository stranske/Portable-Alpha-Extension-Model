from __future__ import annotations

import pandas as pd
import pytest

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator

# ruff: noqa: E402


# Simple pattern to simulate index returns - represents monthly returns
INDEX_SERIES_PATTERN = [0.01, -0.005, 0.008, 0.012, -0.003, 0.006]

EXPECTED = {
    "terminal_AnnReturn": 0.050591712901722374,
    "monthly_AnnVol": 0.014185540196625414,
}


def test_scenario_golden() -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 500,
            "N_MONTHS": 120,
            "w_beta_H": 0.6,
            "w_alpha_H": 0.4,
            "risk_metrics": ["terminal_ShortfallProb"],
        }
    )
    idx = pd.Series(INDEX_SERIES_PATTERN * 20)
    orch = SimulatorOrchestrator(cfg, idx)
    returns, summary = orch.run(seed=0)
    base = summary.loc[summary.Agent == "Base"].iloc[0]
    for key, val in EXPECTED.items():
        assert base[key] == pytest.approx(val, rel=1e-4)
