# ruff: noqa: E402
import numpy as np
import pandas as pd
import pytest

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


def test_te_zero_when_no_alpha() -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 5,
            "N_MONTHS": 12,
            "w_beta_H": 1.0,
            "w_alpha_H": 0.0,
            "external_pa_capital": 0.0,
            "active_ext_capital": 0.0,
            "internal_pa_capital": 0.0,
            "risk_metrics": ["ShortfallProb"],
        }
    )
    idx = pd.Series([0.01, -0.02, 0.015, 0.005] * 3)
    orch = SimulatorOrchestrator(cfg, idx)
    returns, summary = orch.run(seed=0)
    base = returns["Base"]
    internal_beta = returns["InternalBeta"]
    np.testing.assert_allclose(base, internal_beta)
    te_val = summary.loc[summary.Agent == "InternalBeta", "TE"].iloc[0]
    assert te_val == pytest.approx(0.0, abs=1e-12)


def test_orchestrator_reproducible_seed() -> None:
    cfg = load_config(
        {
            "N_SIMULATIONS": 5,
            "N_MONTHS": 6,
            "w_beta_H": 0.6,
            "w_alpha_H": 0.4,
            "risk_metrics": ["ShortfallProb"],
        }
    )
    idx = pd.Series([0.01, 0.02, 0.015, 0.03, 0.005, 0.025])
    orch = SimulatorOrchestrator(cfg, idx)
    ret1, sum1 = orch.run(seed=42)
    ret2, sum2 = orch.run(seed=42)
    for key in ret1:
        if isinstance(ret1[key], pd.DataFrame):
            pd.testing.assert_frame_equal(ret1[key], ret2[key])
        else:
            np.testing.assert_allclose(ret1[key], ret2[key])
    pd.testing.assert_frame_equal(sum1, sum2)
