import pandas as pd
import pytest

from pa_core.config import ModelConfig
from pa_core.sleeve_suggestor import _select_frontier_indices, generate_sleeve_frontier


def _make_linear_summary(cfg: ModelConfig) -> pd.DataFrame:
    per_cap = {
        "ExternalPA": {
            "terminal_AnnReturn": 0.03,
            "terminal_ExcessReturn": 0.025,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
            "terminal_ShortfallProb": 0.0004,
        },
        "ActiveExt": {
            "terminal_AnnReturn": 0.01,
            "terminal_ExcessReturn": 0.008,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
            "terminal_ShortfallProb": 0.0004,
        },
        "InternalPA": {
            "terminal_AnnReturn": 0.02,
            "terminal_ExcessReturn": 0.015,
            "monthly_TE": 0.0001,
            "monthly_BreachProb": 0.0002,
            "monthly_CVaR": -0.0003,
            "terminal_ShortfallProb": 0.0004,
        },
    }
    rows = []
    totals = {
        "terminal_AnnReturn": 0.0,
        "terminal_ExcessReturn": 0.0,
        "monthly_TE": 0.0,
        "monthly_BreachProb": 0.0,
        "monthly_CVaR": 0.0,
        "terminal_ShortfallProb": 0.0,
    }
    for agent, capital in (
        ("ExternalPA", cfg.external_pa_capital),
        ("ActiveExt", cfg.active_ext_capital),
        ("InternalPA", cfg.internal_pa_capital),
    ):
        metrics = {k: v * capital for k, v in per_cap[agent].items()}
        totals = {k: totals[k] + metrics[k] for k in totals}
        rows.append({"Agent": agent, **metrics})
    rows.append({"Agent": "Total", **totals})
    return pd.DataFrame(rows)


def test_frontier_selection_keeps_bin_boundary_points_distinct():
    """Adjacent inclusive risk bins must not spend two frontier slots on one allocation."""
    candidates = pd.DataFrame(
        {
            "constraints_satisfied": [True, True, True],
            "frontier_return": [0.0, 10.0, 1.0],
            "frontier_risk": [0.0, 1.0, 2.0],
        }
    )

    picks = _select_frontier_indices(
        candidates,
        return_col="frontier_return",
        risk_col="frontier_risk",
        min_points=2,
    )

    assert len(picks) == 2
    assert len(set(picks)) == 2
    assert 1 in picks


def test_generate_sleeve_frontier_rejects_negative_evaluation_budget():
    """Frontier generation shares the same non-negative work-budget contract as suggestions."""
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")

    with pytest.raises(ValueError, match="max_evals must be non-negative"):
        generate_sleeve_frontier(
            cfg,
            pd.Series([0.0]),
            max_te=1.0,
            max_breach=1.0,
            max_cvar=1.0,
            max_shortfall=1.0,
            max_evals=-1,
        )


def test_generate_sleeve_frontier_marks_frontier_points(monkeypatch):
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=100.0,
        external_pa_capital=40.0,
        active_ext_capital=30.0,
        internal_pa_capital=30.0,
    )
    idx_series = pd.Series([0.0])

    class DummyOrchestrator:
        def __init__(self, cfg, idx_series):
            self.cfg = cfg
            self.idx_series = idx_series

        def run(self, seed=None):
            return {}, _make_linear_summary(self.cfg)

    monkeypatch.setattr("pa_core.sleeve_suggestor.SimulatorOrchestrator", DummyOrchestrator)

    df = generate_sleeve_frontier(
        cfg,
        idx_series,
        max_te=1.0,
        max_breach=1.0,
        max_cvar=1.0,
        max_shortfall=1.0,
        step=0.1,
        max_evals=None,
        min_frontier_points=20,
    )

    frontier = df[df["is_frontier"]]
    assert len(frontier) == 20
    assert frontier["constraints_satisfied"].all()
