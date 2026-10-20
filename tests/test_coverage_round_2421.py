"""Focused boundary coverage for the ranked #2421 modules."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pa_core import units
from pa_core.llm import result_explain
from pa_core.llm.provider import LLMProviderConfig
from pa_core.sim import internal_pa_financing
from pa_core.sim.metrics import (
    _EXTRA_METRICS,
    _EXTRA_METRIC_DEFINITIONS,
    _cvar_tail_values,
    _load_metric_plugins,
    active_return_volatility,
    annualised_return_percentile,
    breach_count_path0,
    conditional_value_at_risk,
    cvar_confidence_interval,
    cvar_monthly,
    cvar_terminal,
    max_cumulative_sum_drawdown,
    metric_standard_error,
    monte_carlo_convergence_diagnostic,
    per_path_active_return_volatility,
    register_metric,
    terminal_return_below_threshold_prob,
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


def test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StaticLLM:
        def invoke(self, prompt: str) -> Any:
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(result_explain, "create_llm", lambda _config: _StaticLLM())
    monkeypatch.setattr(
        result_explain,
        "record_fleet_event",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fleet down")),
    )

    text, trace_url, payload = result_explain.explain_results_details(
        pd.DataFrame({"value": [1.0]}),
        llm_config=LLMProviderConfig(
            provider_name="openai",
            credentials={"api_key": "sk-test"},
            model_name="gpt-4o-mini",
        ),
        tracing_enabled=False,
    )

    assert text.startswith("ok")
    assert trace_url is None
    assert payload["status"] == "success"
    assert (
        result_explain._extract_metric_delta({"stress_delta_summary": {"summary": "missing"}})
        is None
    )


def test_units_cover_monthly_to_annual_covariance_and_identity_series() -> None:
    covariance = np.eye(2) / 12.0
    assert np.allclose(
        units.convert_covariance(covariance, from_unit="monthly", to_unit="annual"), np.eye(2)
    )

    series = pd.Series([0.01, -0.02], name="returns")
    converted = units.convert_return_series(series, from_unit="monthly", to_unit="monthly")
    assert converted.equals(series)
    assert converted is not series

    monthly = 0.01
    annual_geom = units.convert_mean(
        monthly, from_unit="monthly", to_unit="annual", method="geometric"
    )
    assert annual_geom == pytest.approx((1.0 + monthly) ** 12 - 1.0)
    assert units.convert_return_series(
        series, from_unit="monthly", to_unit="annual", method="geometric"
    ).tolist() == pytest.approx(((1.0 + series) ** 12 - 1.0).tolist())


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


def test_metrics_cover_registration_plugins_and_boundary_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "CoverageRound2421_DuplicateMetric"

    def _metric(arr: np.ndarray) -> float:
        return float(np.mean(arr))

    monkeypatch.setitem(_EXTRA_METRICS, name, _metric)
    monkeypatch.setitem(_EXTRA_METRIC_DEFINITIONS, name, {})
    _EXTRA_METRICS.pop(name)
    _EXTRA_METRIC_DEFINITIONS.pop(name)
    register_metric(name, _metric)
    with pytest.raises(KeyError, match="already registered"):
        register_metric(name, _metric)

    loaded: list[str] = []

    class _EP:
        def __init__(self, ep_name: str) -> None:
            self.name = ep_name

        def load(self) -> Any:
            loaded.append(self.name)
            return _metric

    monkeypatch.setattr(
        "pa_core.sim.metrics.entry_points",
        lambda group="pa_core.risk_metrics": [_EP("CoverageRound2421_PluginMetric")],
    )
    plugin_name = "CoverageRound2421_PluginMetric"
    monkeypatch.setitem(_EXTRA_METRICS, plugin_name, _metric)
    monkeypatch.setitem(_EXTRA_METRIC_DEFINITIONS, plugin_name, {})
    _EXTRA_METRICS.pop(plugin_name)
    _EXTRA_METRIC_DEFINITIONS.pop(plugin_name)
    _load_metric_plugins()
    assert loaded == ["CoverageRound2421_PluginMetric"]
    assert plugin_name in _EXTRA_METRICS
    assert _EXTRA_METRICS[plugin_name] is _metric
    assert _EXTRA_METRIC_DEFINITIONS[plugin_name] == {
        "Metric": plugin_name,
        "MetricType": "diagnostic",
        "Description": "User-registered metric.",
    }

    strategy = np.array([0.01, 0.02, -0.01])
    benchmark = np.array([0.0, 0.01, 0.0])
    assert per_path_active_return_volatility(strategy, benchmark) == pytest.approx(
        active_return_volatility(strategy, benchmark)
    )

    # Single-month 1D input forces the window==1 branch; -2% is below the
    # horizon-scaled annual hurdle for one month, so shortfall probability is 1.
    shortfall = terminal_return_below_threshold_prob(
        np.array([-0.02]), threshold=-0.05, periods_per_year=12
    )
    assert shortfall == pytest.approx(1.0)
    assert terminal_return_below_threshold_prob(
        np.array([0.0]), threshold=-0.05, periods_per_year=12
    ) == pytest.approx(0.0)

    one_path = np.array([0.01, -0.02, 0.015])
    terminal = float(np.prod(1.0 + one_path) - 1.0)
    assert cvar_terminal(one_path) == pytest.approx(terminal)

    diag = monte_carlo_convergence_diagnostic(one_path, metric="monthly_CVaR", checkpoints=(1.0,))
    assert list(diag["metric"]) == ["monthly_CVaR"]
    assert diag["estimate"].iloc[0] == pytest.approx(cvar_monthly(one_path))
    ann = monte_carlo_convergence_diagnostic(
        one_path, metric="terminal_AnnReturn", checkpoints=(1.0,)
    )
    assert list(ann["metric"]) == ["terminal_AnnReturn"]
    # Three monthly returns annualize over four independent repetitions.
    expected_annual = (1.01 * 0.98 * 1.015) ** 4 - 1.0
    assert ann["estimate"].iloc[0] == pytest.approx(expected_annual)

    with pytest.raises(ValueError, match="returns must not be empty"):
        max_cumulative_sum_drawdown(np.array([]))
    assert max_cumulative_sum_drawdown(np.array([0.1, -0.05])) < 0.0
