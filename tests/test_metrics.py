import numpy as np
import pandas as pd

from pa_core.sim.metrics import (
    annualised_return,
    annualised_vol,
    breach_count,
    breach_probability,
    compound,
    conditional_value_at_risk,
    max_drawdown,
    shortfall_probability,
    summary_table,
    time_under_water,
    tracking_error,
    value_at_risk,
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


def test_summary_table_adds_total_when_benchmark_present():
    base = np.zeros((2, 3))
    ext = np.array([[0.01, -0.01, 0.02], [0.0, 0.01, -0.02]])
    ibeta = np.array([[0.005, 0.0, -0.005], [0.01, -0.005, 0.0]])
    returns = {"Base": base, "ExternalPA": ext, "InternalBeta": ibeta}

    stats = summary_table(returns, benchmark="Base")
    assert "Total" in stats["Agent"].values

    total = ext + ibeta
    stats_with_total = summary_table({**returns, "Total": total}, benchmark="Base")

    total_row = stats[stats["Agent"] == "Total"].iloc[0]
    total_row_explicit = stats_with_total[stats_with_total["Agent"] == "Total"].iloc[0]
    for col in ["AnnReturn", "AnnVol", "VaR", "CVaR", "BreachProb", "ShortfallProb", "TE"]:
        val = total_row[col]
        expected = total_row_explicit[col]
        if pd.isna(val) and pd.isna(expected):
            continue
        assert np.isclose(float(val), float(expected))


def test_conditional_value_at_risk_monotonic():
    arr1 = np.array([-0.01] * 100)
    arr2 = np.array([-0.01] * 99 + [-0.5])
    cvar1 = conditional_value_at_risk(arr1, 0.95)
    cvar2 = conditional_value_at_risk(arr2, 0.95)
    assert cvar2 <= cvar1


def test_max_drawdown_basic():
    arr = np.array([[0.01, -0.02, 0.03]])
    dd = max_drawdown(arr)
    assert dd <= 0
    pos = np.array([[0.01, 0.02, 0.03]])
    assert max_drawdown(pos) == 0.0


def test_time_under_water_basic():
    arr = np.array([[0.01, -0.02, 0.01]])
    tuw = time_under_water(arr)
    assert 0 < tuw < 1
    pos = np.array([[0.01, 0.02]])
    assert time_under_water(pos) == 0.0


def test_breach_count_basic():
    arr = np.array([[0.0, -0.05, 0.01]])
    assert breach_count(arr, -0.01) == 1


def test_breach_probability_basic():
    arr = np.array([[0.0, -0.05, 0.01], [0.02, 0.01, 0.03]])
    threshold = -0.01
    prob = breach_probability(arr, threshold)
    assert np.isclose(prob, 1.0 / 6.0)


def test_breach_probability_modes_multi_path():
    arr = np.array([[0.0, -0.05, 0.01], [0.02, 0.01, 0.03]])
    threshold = -0.01
    assert np.isclose(breach_probability(arr, threshold, mode="any"), 0.5)
    assert np.isclose(breach_probability(arr, threshold, mode="terminal"), 0.0)


def test_breach_probability_single_path():
    arr = np.array([0.0, -0.05, 0.01])
    thr = -0.01
    assert np.isclose(breach_probability(arr, thr), 1.0 / 3.0)


def test_breach_probability_modes_single_path():
    arr = np.array([0.0, -0.05, 0.01])
    thr = -0.01
    assert np.isclose(breach_probability(arr, thr, mode="any"), 1.0)
    assert np.isclose(breach_probability(arr, thr, mode="terminal"), 0.0)


def test_breach_probability_ignores_path_argument():
    arr = np.array([[0.0, -0.05, 0.01], [-0.02, 0.02, 0.03]])
    thr = -0.01
    expected = 2.0 / 6.0
    assert np.isclose(breach_probability(arr, thr, path=0), expected)
    assert np.isclose(breach_probability(arr, thr, path=1), expected)


def test_summary_table_breach():
    arr = np.array([[0.0, -0.03, 0.03], [0.01, 0.02, 0.03]])
    stats = summary_table({"Base": arr})
    assert "BreachProb" in stats.columns
    assert np.isclose(stats["BreachProb"].iloc[0], 1.0 / 6.0)


def test_summary_table_breach_custom():
    arr = np.array([[0.0, -0.02, 0.03], [0.01, 0.02, 0.03]])
    stats = summary_table({"Base": arr}, breach_threshold=-0.01)
    assert np.isclose(stats["BreachProb"].iloc[0], 1.0 / 6.0)


def test_breach_probability_path_order_invariant():
    arr = np.array(
        [
            [0.0, -0.02, 0.01],
            [0.01, 0.02, 0.03],
            [-0.03, 0.02, 0.01],
        ]
    )
    thr = -0.01
    prob = breach_probability(arr, thr)
    reversed_prob = breach_probability(arr[::-1], thr)
    assert prob == reversed_prob


def test_breach_probability_reproducible_seed():
    rng = np.random.default_rng(123)
    arr = rng.normal(0.0, 0.1, size=(100, 12))
    prob = breach_probability(arr, -0.02)
    rng = np.random.default_rng(123)
    arr2 = rng.normal(0.0, 0.1, size=(100, 12))
    prob2 = breach_probability(arr2, -0.02)
    assert prob == prob2


def test_breach_probability_empty_input():
    try:
        breach_probability(np.array([]), -0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty returns")


def test_breach_probability_invalid_mode():
    arr = np.array([0.0, -0.05, 0.01])
    try:
        breach_probability(arr, -0.01, mode="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid mode")


def test_summary_table_includes_new_metrics():
    arr = np.array([[0.0, -0.03, 0.03]])
    stats = summary_table({"Base": arr})
    for col in {"CVaR", "MaxDD", "TimeUnderWater", "BreachCount"}:
        assert col in stats.columns


def test_shortfall_probability_basic():
    arr = np.array([[0.1, -0.2], [0.05, 0.02]])
    prob = shortfall_probability(arr, threshold=-0.05)
    assert prob == 0.5


def test_summary_table_shortfall():
    arr = np.array([[0.1, -0.2], [0.05, 0.02]])
    stats = summary_table({"A": arr})
    assert "ShortfallProb" in stats.columns
    assert stats["ShortfallProb"].iloc[0] == 0.5
