import numpy as np
from pa_core.metrics import (
    tracking_error,
    value_at_risk,
    compound,
    annualised_return,
    annualised_vol,
    breach_probability,
    summary_table,
)


def test_tracking_error_constant_series():
    strat = np.full(12, 0.05)
    bench = np.full(12, 0.05)
    assert tracking_error(strat, bench) == 0.0


def test_value_at_risk_constant_series():
    returns = np.full(20, -0.01)
    assert value_at_risk(returns, 0.95) == -0.01


def test_value_at_risk_extreme():
    returns = np.array([-0.5] * 10 + [0.1] * 90)
    var = value_at_risk(returns, 0.99)
    assert var <= -0.5


def test_metrics_shape_mismatch():
    try:
        tracking_error(np.arange(5), np.arange(6))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_compound_and_summary():
    arr = np.array([[0.01, -0.02, 0.03]])
    comp = compound(arr)
    assert comp.shape == arr.shape
    ann_ret = annualised_return(arr)
    ann_vol = annualised_vol(arr)
    stats = summary_table({"Base": arr})
    assert "AnnReturn" in stats.columns
    assert np.isfinite(ann_ret)
    assert np.isfinite(ann_vol)


def test_breach_probability_basic():
    arr = np.array([[0.0, -0.05, 0.01]])
    threshold = -0.01
    prob = breach_probability(arr, threshold)
    assert prob == 1 / 3


def test_summary_table_breach():
    arr = np.array([[0.0, -0.02, 0.03]])
    stats = summary_table({"Base": arr}, breach_threshold=-0.01)
    assert "BreachProb" in stats.columns

