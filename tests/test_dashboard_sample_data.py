"""Tests for the bundled sample-data onboarding affordance (issue #1900).

The dashboard previously gated every data-driven page behind a file upload and
never auto-surfaced the bundled ``data/sp500tr_fred_divyield.csv`` series, so a
first-run user saw only empty placeholders. These tests cover the shared helper
that exposes the bundled series and verify it runs end-to-end through the
simulator, which is what the "Use bundled sample data" one-click affordance relies
on.
"""

from __future__ import annotations

import pandas as pd

from dashboard.utils import (
    SAMPLE_INDEX_FILENAME,
    SAMPLE_FINANCING_MODE,
    build_sample_model_config,
    bundled_sample_index_path,
    load_bundled_sample_index,
)
from pa_core.config import ModelConfig
from pa_core.data import load_index_returns
from pa_core.orchestrator import SimulatorOrchestrator


def test_bundled_sample_index_path_points_at_repo_data_file() -> None:
    path = bundled_sample_index_path()
    assert path.exists(), f"bundled sample dataset missing at {path}"
    assert path.name == SAMPLE_INDEX_FILENAME
    assert path.parent.name == "data"


def test_load_bundled_sample_index_returns_non_empty_numeric_series() -> None:
    series = load_bundled_sample_index()
    expected = load_index_returns(bundled_sample_index_path())
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert pd.api.types.is_numeric_dtype(series)
    assert not series.isna().any()
    assert series.equals(expected)


def test_bundled_sample_runs_through_orchestrator_end_to_end() -> None:
    """The bundled series must produce a real summary (the one-click demo claim)."""
    index_series = load_bundled_sample_index()
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 100,
            "Number of months": 12,
            "financing_mode": "broadcast",
        }
    )
    orch = SimulatorOrchestrator(cfg, index_series)
    _, summary = orch.run(seed=42)
    assert summary is not None
    assert not summary.empty


def test_sample_run_config_validates() -> None:
    """Stress Lab and Scenario Grid sample config paths must not dead-end."""
    stress_config = build_sample_model_config(
        **{
            "Number of simulations": 1000,
            "Number of months": 12,
            "Total fund capital (mm)": 1000.0,
            "External PA capital (mm)": 200.0,
            "Active Extension capital (mm)": 200.0,
            "Internal PA capital (mm)": 200.0,
            "External PA alpha fraction": 0.5,
            "Active share": 0.5,
        }
    )
    grid_config = build_sample_model_config()

    assert stress_config.financing_mode == SAMPLE_FINANCING_MODE
    assert grid_config.financing_mode == SAMPLE_FINANCING_MODE
    assert stress_config.financing_mode in {"per_path", "broadcast"}
    assert grid_config.financing_mode in {"per_path", "broadcast"}
