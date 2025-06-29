import numpy as np
from pa_core.metrics import tracking_error, value_at_risk


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

