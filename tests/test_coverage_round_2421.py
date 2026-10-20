"""Focused boundary coverage for the ranked #2421 modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_core import units
from pa_core.llm import result_explain
from pa_core.sim import internal_pa_financing
from pa_core.sim.metrics import (
    _cvar_tail_values,
    active_return_volatility,
    annualised_return_percentile,
    breach_count_path0,
    conditional_value_at_risk,
    cvar_confidence_interval,
    cvar_terminal,
    metric_standard_error,
    monte_carlo_convergence_diagnostic,
    per_path_active_return_volatility,
    value_at_risk,
)


def test_result_helpers_preserve_json_lists_and_prefer_total_metric_row() -> None:
    value = [pd.Timestamp("2026-01-01"), pd.NA]
    assert result_explain._to_json_safe(value) == ["2026-01-01T00:00:00", None]

    details = pd.DataFrame({"Agent": ["Strategy", "Total"], "Monthly TE": [0.01, 0.02]})
    column = result_explain._find_metric_column(details, ("monthly_te",))
    assert column == "Monthly TE"
    assert result_explain._coerce_metric_value(details, column) == pytest.approx(0.02)
    assert result_explain._find_metric_column(details, ("not a metric",)) is None


def test_units_cover_monthly_to_annual_covariance_and_identity_series() -> None:
    covariance = np.eye(2) / 12.0
    assert np.allclose(
        units.convert_covariance(covariance, from_unit="monthly", to_unit="annual"), np.eye(2)
    )

    series = pd.Series([0.01, -0.02], name="returns")
    converted = units.convert_return_series(series, from_unit="monthly", to_unit="monthly")
    assert converted.equals(series)
    assert converted is not series


def test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_rng = np.random.default_rng(7)
    calls: list[tuple[object, int]] = []

    def fake_spawn_rngs(seed: object, count: int) -> list[np.random.Generator]:
        calls.append((seed, count))
        return [expected_rng]

    monkeypatch.setattr(internal_pa_financing, "spawn_rngs", fake_spawn_rngs)

    result = internal_pa_financing.resolve_internal_pa_financing_series(
        n_months=2, n_sim=2, sigma_month=0.01, financing_mode="broadcast"
    )

    assert calls == [(None, 1)]
    assert result.shape == (2, 2)
    assert np.allclose(result[0], result[1])


def test_metrics_reject_invalid_inputs_and_handle_small_samples() -> None:
    with pytest.raises(ValueError, match="confidence"):
        value_at_risk(np.array([0.01]), confidence=1.0)
    with pytest.raises(ValueError, match="confidence"):
        conditional_value_at_risk(np.array([0.01]), confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        _cvar_tail_values(np.array([0.01]), confidence=1.0)
    with pytest.raises(ValueError, match="returns must not be empty"):
        _cvar_tail_values(np.array([]))
    with pytest.raises(ValueError, match="interval_confidence"):
        cvar_confidence_interval(np.array([0.01, -0.01]), interval_confidence=1.0)

    assert active_return_volatility(np.array([0.01]), np.array([0.0])) == 0.0
    assert per_path_active_return_volatility(np.array([[0.01], [0.02]]), np.zeros((2, 1))) == 0.0
    assert breach_count_path0(np.array([0.01, -0.02]), 0.0) == 1
    with pytest.raises(IndexError, match="path index"):
        breach_count_path0(np.array([[0.01]]), 0.0, path=1)
    with pytest.raises(ValueError, match="1D or 2D"):
        annualised_return_percentile(np.zeros((1, 1, 1)), 50)
    with pytest.raises(ValueError, match="returns must not be empty"):
        cvar_terminal(np.array([]))
    assert metric_standard_error(np.array([0.01])) == 0.0
    with pytest.raises(ValueError, match="values must not be empty"):
        metric_standard_error(np.array([]))


def test_convergence_diagnostic_rejects_invalid_requests() -> None:
    returns = np.array([[0.01, -0.02], [0.02, -0.01]])
    with pytest.raises(ValueError, match="returns must not be empty"):
        monte_carlo_convergence_diagnostic(np.array([]))
    with pytest.raises(ValueError, match="1D or 2D"):
        monte_carlo_convergence_diagnostic(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="checkpoints must not be empty"):
        monte_carlo_convergence_diagnostic(returns, checkpoints=())
    with pytest.raises(ValueError, match="interval"):
        monte_carlo_convergence_diagnostic(returns, checkpoints=(0.0,))
    with pytest.raises(ValueError, match="metric must be one of"):
        monte_carlo_convergence_diagnostic(returns, metric="unsupported")
