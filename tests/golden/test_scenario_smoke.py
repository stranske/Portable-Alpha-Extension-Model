from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator


INDEX_SERIES_PATTERN = [0.01, 0.02, 0.015, 0.03, 0.005, 0.025]
EXPECTED = {
    "AnnReturn": 0.026798836068948395,
    "AnnVol": 0.0066524784538782465,
}


def test_scenario_golden() -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 500,
            "N_MONTHS": 120,
            "w_beta_H": 0.6,
            "w_alpha_H": 0.4,
            "risk_metrics": ["ShortfallProb"],
        }
    )
    idx = pd.Series(INDEX_SERIES_PATTERN * 20)
    orch = SimulatorOrchestrator(cfg, idx)
    returns, summary = orch.run(seed=0)
    base = summary.loc[summary.Agent == "Base"].iloc[0]
    for key, val in EXPECTED.items():
        assert base[key] == pytest.approx(val, rel=1e-4)
