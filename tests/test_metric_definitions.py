import numpy as np
import pytest

from pa_core.agents.risk_metrics import RiskMetricsAgent
from pa_core.sim.metrics import (
    breach_probability,
    conditional_value_at_risk,
    summary_table,
    terminal_return_below_threshold_prob,
    value_at_risk,
)


def test_cvar_strict_tail_excludes_var_cutoff() -> None:
    arr = np.array([-0.10, -0.05, -0.05, 0.02])
    var = value_at_risk(arr, confidence=0.5)
    cvar = conditional_value_at_risk(arr, confidence=0.5)
    assert var == pytest.approx(-0.05)
    assert cvar == pytest.approx(-0.10)


def test_cvar_empty_tail_falls_back_to_var() -> None:
    arr = np.array([-0.02, -0.02, -0.02])
    var = value_at_risk(arr, confidence=0.95)
    cvar = conditional_value_at_risk(arr, confidence=0.95)
    assert cvar == pytest.approx(var)


def test_breach_probability_all_paths_all_months() -> None:
    arr = np.array([[-0.01, 0.02, -0.03], [0.01, -0.04, 0.02]])
    prob = breach_probability(arr, threshold=-0.02)
    assert prob == pytest.approx(2.0 / 6.0)


def test_breach_probability_consistent_across_callers() -> None:
    arr = np.array([[-0.01, 0.02, -0.03], [0.01, -0.04, 0.02]])
    threshold = -0.02
    expected = breach_probability(arr, threshold=threshold)
    agent = RiskMetricsAgent(breach_threshold=threshold)
    metrics = agent.run(arr)
    table = summary_table({"A": arr}, breach_threshold=threshold)
    assert metrics.breach_probability == pytest.approx(expected)
    assert float(table.loc[0, "BreachProb"]) == pytest.approx(expected)


def test_shortfall_threshold_annual_to_horizon_adjustment() -> None:
    arr = np.zeros((2, 24))
    arr[0, 0] = -0.15
    arr[1, 0] = -0.25
    prob = terminal_return_below_threshold_prob(arr, threshold=-0.1, periods_per_year=12)
    assert prob == pytest.approx(0.5)
