"""Focused behavioral coverage for scenario-comparison plotting helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

module = importlib.import_module("pa_core.viz.compare_scenarios")


def _summary(agent: str = "Base") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Agent": [agent],
            "terminal_AnnReturn": [0.08],
            "monthly_AnnVol": [0.12],
            "monthly_CVaR": [-0.03],
            "monthly_VaR": [-0.02],
            "monthly_MaxDD": [-0.11],
        }
    )


def test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs() -> None:
    scenarios = [
        {
            "combination_id": 7,
            "summary": _summary(),
            "raw_returns": {
                "Base": pd.DataFrame([[0.01, -0.02], [0.03, 0.01]]),
                "Regime": [["normal", "stress"], ["normal", "normal"]],
            },
        },
        {
            "label": "Alternative",
            "summary": _summary("Total"),
            "returns": [0.01, 0.02, -0.01],
            "regime_labels": {"Regime": ["stress", "normal", "normal"]},
        },
    ]

    figures = module.compare_scenarios(scenarios)

    assert set(figures) == {
        "risk_return",
        "cvar_return",
        "return_distribution",
        "risk_metrics",
        "regime_timeline",
    }
    assert figures["cvar_return"].data[0].text.tolist() == ["Scenario 7", "Alternative"]
    assert len(figures["return_distribution"].data) == 4
    assert len(figures["regime_timeline"].data) > 0


def test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs() -> None:
    figures = module.compare_scenarios([_summary()], include_returns=False)
    assert set(figures) == {"risk_return", "cvar_return", "risk_metrics"}

    with pytest.raises(ValueError, match="at least one"):
        module.compare_scenarios([])
    with pytest.raises(TypeError, match="summary"):
        module._extract_summary({"summary": "not-a-frame"})
    with pytest.raises(TypeError, match="returns"):
        module._extract_returns({})
    with pytest.raises(KeyError, match="No return series"):
        module._select_returns_key({"empty": None})
    with pytest.raises(KeyError, match="No CVaR"):
        module._pick_column(pd.DataFrame({"x": [1]}), {}, "CVaR")
    with pytest.raises(KeyError, match="Missing risk metric"):
        module._make_risk_metric_bars(pd.DataFrame({"Agent": ["Base"]}))


def test_distribution_helpers_handle_empty_constant_and_large_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        module._terminal_compounded_returns(np.array([]))
    with pytest.raises(ValueError, match="no finite"):
        module._terminal_compounded_returns(np.array([[np.nan, np.nan]]))

    grid = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(module._kde_density(np.array([1.0]), grid), np.zeros_like(grid))
    assert np.array_equal(module._kde_density(np.array([1.0, 1.0]), grid), np.zeros_like(grid))

    samples = np.arange(6_000, dtype=float)
    downsampled = module._downsample(samples, max_samples=100)
    assert downsampled.size == 100
    assert downsampled[0] == 0
    assert downsampled[-1] == 5_999
    assert np.all(module._kde_density(samples, grid) >= 0)
