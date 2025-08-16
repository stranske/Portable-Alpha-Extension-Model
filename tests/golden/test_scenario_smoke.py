from __future__ import annotations

# ruff: noqa: E402

import types
import sys
from pathlib import Path

import pandas as pd
import pytest

# Prevent importing full pa_core package with heavy deps
PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator


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
