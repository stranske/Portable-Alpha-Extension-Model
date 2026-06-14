import numpy as np
import pandas as pd
import pytest

from pa_core.sim.metrics import (
    active_return_volatility,
    annualised_return,
    annualised_return_percentile,
    annualised_vol,
    breach_count,
    breach_count_path0,
    breach_probability,
    compound,
    compounded_return_below_zero_fraction,
    conditional_value_at_risk,
    cvar_terminal,
    max_cumulative_sum_drawdown,
    max_drawdown,
    per_path_active_return_volatility,
    shortfall_probability,
    summary_table,
    terminal_return_below_threshold_prob,
    time_under_water,
    tracking_error,
    value_at_risk,
)
from pa_core.sim.paths import prepare_mc_universe


def test_active_return_volatility_constant_series():
    strat = np.full(12, 0.05)
    bench = np.full(12, 0.05)
    assert active_return_volatility(strat, bench) == 0.0


def test_value_at_risk_constant_series():
    returns = np.full(20, -0.01)
    assert value_at_risk(returns, 0.95) == -0.01


def test_value_at_risk_extreme():
    returns = np.array([-0.5] * 10 + [0.1] * 90)
    var = value_at_risk(returns, 0.99)
    assert var <= -0.5


def test_metrics_shape_mismatch():
    try:
        active_return_volatility(np.arange(5), np.arange(6))
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
    assert "terminal_AnnReturn" in stats.columns
    assert np.isfinite(ann_ret)
    assert np.isfinite(ann_vol)


def test_annualised_return_total_loss_is_finite():
    # A catastrophic month (<= -100%) drives the mean terminal multiple to <= 0
    # (base 1 + mean <= 0); this must not produce NaN / RuntimeWarning from a
    # fractional power of a non-positive base.
    arr = np.array([[-1.5, 0.2]])
    ann_ret = annualised_return(arr)
    assert np.isfinite(ann_ret)
    assert ann_ret == -1.0


def test_annualised_vol_single_observation_is_finite():
    # ddof=1 sample std is undefined for a single observation; guard returns 0.0
    # rather than NaN from the divide-by-zero.
    arr = np.array([[0.01]])
    ann_vol = annualised_vol(arr)
    assert np.isfinite(ann_vol)
    assert ann_vol == 0.0


def test_annualised_vol_rejects_empty_input():
    with pytest.raises(ValueError, match="returns must not be empty"):
        annualised_vol(np.array([]))


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
    for col in [
        "terminal_AnnReturn",
        "monthly_AnnVol",
        "monthly_VaR",
        "monthly_CVaR",
        "terminal_CVaR",
        "monthly_BreachProb",
        "terminal_ShortfallProb",
        "monthly_TE",
    ]:
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


def test_terminal_cvar_worsens_with_heavier_tails_t_copula():
    cov = np.eye(4) * (0.03**2)
    low_df = prepare_mc_universe(
        N_SIMULATIONS=3000,
        N_MONTHS=24,
        mu_idx=0.0,
        mu_H=0.0,
        mu_E=0.0,
        mu_M=0.0,
        cov_mat=cov,
        return_distribution="student_t",
        return_t_df=4.0,
        return_copula="t",
        seed=123,
    )
    high_df = prepare_mc_universe(
        N_SIMULATIONS=3000,
        N_MONTHS=24,
        mu_idx=0.0,
        mu_H=0.0,
        mu_E=0.0,
        mu_M=0.0,
        cov_mat=cov,
        return_distribution="student_t",
        return_t_df=30.0,
        return_copula="t",
        seed=123,
    )
    low_tail_cvar = cvar_terminal(low_df[:, :, 0], confidence=0.95)
    high_tail_cvar = cvar_terminal(high_df[:, :, 0], confidence=0.95)
    assert low_tail_cvar <= high_tail_cvar


def test_max_cumulative_sum_drawdown_basic():
    arr = np.array([[0.01, -0.02, 0.03]])
    dd = max_cumulative_sum_drawdown(arr)
    assert np.isclose(dd, -0.02)
    pos = np.array([[0.01, 0.02, 0.03]])
    assert max_cumulative_sum_drawdown(pos) == 0.0


def test_max_cumulative_sum_drawdown_compounded_path():
    arr = np.array([[0.2, -0.1, -0.1]])
    dd = max_cumulative_sum_drawdown(arr)
    assert np.isclose(dd, -0.19)


def test_max_drawdown_uses_compounded_path():
    arr = np.array([[0.2, -0.1, -0.1]])
    dd = max_drawdown(arr)
    assert np.isclose(dd, -0.19)


def test_max_cumulative_sum_drawdown_initial_drop():
    arr = np.array([[-0.1, 0.0, 0.05]])
    dd = max_cumulative_sum_drawdown(arr)
    assert np.isclose(dd, -0.1)


def test_compounded_return_below_zero_fraction_basic():
    arr = np.array([[0.01, -0.02, 0.01]])
    tuw = compounded_return_below_zero_fraction(arr)
    assert 0 < tuw < 1
    pos = np.array([[0.01, 0.02]])
    assert compounded_return_below_zero_fraction(pos) == 0.0


def test_breach_count_basic():
    arr = np.array([[0.0, -0.05, 0.01]])
    assert breach_count_path0(arr, -0.01) == 1


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
    assert "monthly_BreachProb" in stats.columns
    assert np.isclose(stats["monthly_BreachProb"].iloc[0], 1.0 / 6.0)


def test_summary_table_breach_custom():
    arr = np.array([[0.0, -0.02, 0.03], [0.01, 0.02, 0.03]])
    stats = summary_table({"Base": arr}, breach_threshold=-0.01)
    assert np.isclose(stats["monthly_BreachProb"].iloc[0], 1.0 / 6.0)


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
    for col in {
        "monthly_CVaR",
        "terminal_CVaR",
        "monthly_MaxDD",
        "monthly_TimeUnderWater",
        "monthly_BreachCountPath0",
    }:
        assert col in stats.columns


def test_terminal_return_below_threshold_prob_basic():
    arr = np.array([[0.1, -0.2], [0.05, 0.02]])
    prob = terminal_return_below_threshold_prob(arr, threshold=-0.05)
    assert prob == 0.5


def test_terminal_return_below_threshold_prob_rolling_window():
    arr = np.array([0.1, -0.2, 0.05, 0.02])
    prob = terminal_return_below_threshold_prob(arr, threshold=-0.05, periods_per_year=2)
    assert np.isclose(prob, 2.0 / 3.0)


def test_terminal_return_below_threshold_prob_horizon_threshold():
    arr = np.zeros((2, 12))
    prob = terminal_return_below_threshold_prob(arr, threshold=-0.12, periods_per_year=12)
    assert prob == 0.0


def test_terminal_return_below_threshold_prob_empty_input():
    try:
        terminal_return_below_threshold_prob(np.array([]), threshold=-0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty returns")


def test_active_return_volatility_annualised():
    strat = np.array([0.02, -0.02, 0.0, 0.01])
    bench = np.zeros_like(strat)
    te = active_return_volatility(strat, bench, periods_per_year=12)
    expected = np.std(strat, ddof=1) * np.sqrt(12)
    assert np.isclose(te, expected)


def test_summary_table_shortfall():
    arr = np.array([[0.1, -0.2], [0.05, 0.02]])
    stats = summary_table({"A": arr})
    assert "terminal_ShortfallProb" in stats.columns
    assert stats["terminal_ShortfallProb"].iloc[0] == 0.5


def test_annualised_return_percentile_orders_p10_median_p90():
    # Wide spread of single-month terminal returns across paths.
    arr = np.array([[-0.3], [-0.1], [0.0], [0.1], [0.4]])
    p10 = annualised_return_percentile(arr, 10, periods_per_year=12)
    p50 = annualised_return_percentile(arr, 50, periods_per_year=12)
    p90 = annualised_return_percentile(arr, 90, periods_per_year=12)
    assert p10 < p50 < p90
    # Median path here has a 0.0 terminal return -> 0.0 annualised.
    assert p50 == pytest.approx(0.0)


def test_annualised_return_percentile_matches_annualise_of_percentile():
    arr = np.array([[0.02, -0.01, 0.03], [0.01, 0.0, -0.02], [0.05, 0.01, 0.0]])
    comp = compound(arr)
    total = comp[:, -1]
    years = arr.shape[1] / 12
    expected = (1.0 + float(np.percentile(total, 50))) ** (1.0 / years) - 1.0
    assert annualised_return_percentile(arr, 50) == pytest.approx(expected)


def test_annualised_return_percentile_total_loss_floor():
    # A percentile terminal multiple wiped out should floor at -1.0, not raise.
    arr = np.array([[-1.0], [-1.0], [-1.0]])
    assert annualised_return_percentile(arr, 50, periods_per_year=12) == -1.0


def test_annualised_return_percentile_accepts_single_path_series():
    arr = np.array([0.02, -0.01, 0.03])
    expected = annualised_return_percentile(arr.reshape(1, -1), 50)
    assert annualised_return_percentile(arr, 50) == pytest.approx(expected)


def test_annualised_return_percentile_rejects_empty_input():
    with pytest.raises(ValueError, match="returns must not be empty"):
        annualised_return_percentile(np.array([]), 50)


def test_summary_table_includes_terminal_return_percentiles():
    arr = np.array([[-0.2], [0.0], [0.3]])
    stats = summary_table({"Base": arr})
    for col in ("terminal_AnnReturn_P50", "terminal_AnnReturn_P10", "terminal_AnnReturn_P90"):
        assert col in stats.columns
    row = stats.iloc[0]
    assert row["terminal_AnnReturn_P10"] <= row["terminal_AnnReturn_P50"] <= row["terminal_AnnReturn_P90"]


def test_per_path_te_lower_than_pooled_when_means_differ_across_paths():
    # Two paths with no within-path active variation but different active means:
    # pooled TE picks up the cross-path mean spread, per-path TE is ~0.
    strat = np.array([[0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]])
    bench = np.zeros_like(strat)
    pooled = active_return_volatility(strat, bench, periods_per_year=12)
    per_path = per_path_active_return_volatility(strat, bench, periods_per_year=12)
    assert per_path == pytest.approx(0.0)
    assert pooled > per_path


def test_per_path_te_matches_active_vol_for_single_path():
    strat = np.array([[0.02, -0.02, 0.0, 0.01]])
    bench = np.zeros_like(strat)
    assert per_path_active_return_volatility(strat, bench) == pytest.approx(
        active_return_volatility(strat, bench)
    )


def test_per_path_te_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_path_active_return_volatility(np.zeros((2, 3)), np.zeros((2, 4)))


def test_summary_table_includes_per_path_te_with_benchmark():
    returns = {
        "Base": np.array([[0.01, -0.02, 0.03], [0.0, 0.01, -0.01]]),
        "Active": np.array([[0.03, -0.01, 0.04], [0.01, 0.02, 0.0]]),
    }
    stats = summary_table(returns, benchmark="Base").set_index("Agent")
    assert "monthly_TE_PerPath" in stats.columns
    # Benchmark agent has no tracking error; active agent has a finite per-path TE.
    assert pd.isna(stats.loc["Base", "monthly_TE_PerPath"])
    assert np.isfinite(stats.loc["Active", "monthly_TE_PerPath"])


def test_deprecated_metric_aliases_warn():
    arr = np.array([[0.1, -0.2, 0.05]])
    with pytest.warns(DeprecationWarning, match="tracking_error"):
        tracking_error(np.array([0.01, 0.02]), np.array([0.0, 0.0]))
    with pytest.warns(DeprecationWarning, match="max_drawdown"):
        max_drawdown(arr)
    with pytest.warns(DeprecationWarning, match="time_under_water"):
        time_under_water(arr)
    with pytest.warns(DeprecationWarning, match="shortfall_probability"):
        shortfall_probability(arr, threshold=-0.05)
    with pytest.warns(DeprecationWarning, match="breach_count"):
        breach_count(arr, threshold=-0.05)
