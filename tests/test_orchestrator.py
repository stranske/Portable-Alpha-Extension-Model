# ruff: noqa: E402
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator


def test_orchestrator_runs() -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 10,
            "N_MONTHS": 5,
            "w_beta_H": 0.6,
            "w_alpha_H": 0.4,
            "risk_metrics": ["ShortfallProb"],
        }
    )
    idx = pd.Series([0.01, 0.02, 0.015, 0.03, 0.005, 0.025] * 20)
    orch = SimulatorOrchestrator(cfg, idx)
    returns, summary = orch.run(seed=0)
    assert "Base" in returns
    assert "AnnReturn" in summary.columns


def test_invalid_share_sum() -> None:
    with pytest.raises(ValueError):
        load_config(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "w_beta_H": 0.7,
                "w_alpha_H": 0.4,
                "risk_metrics": ["ShortfallProb"],
            }
        )
