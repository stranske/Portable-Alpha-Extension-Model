"""Tests for the bundled sample-data onboarding affordance (issue #1900).

The dashboard previously gated every data-driven page behind a file upload and
never auto-surfaced the bundled ``data/sp500tr_fred_divyield.csv`` series, so a
first-run user saw only empty placeholders. These tests cover the shared helper
that exposes the bundled series and verify it runs end-to-end through the
simulator, which is what the "Use bundled sample data" one-click affordance relies
on.
"""

from __future__ import annotations

import subprocess
import venv
from pathlib import Path

import pandas as pd
import pytest

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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(destination: Path) -> Path:
    environment = destination / "wheel-build-env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "setuptools>=82.0.1", "wheel"],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_dir = destination / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "-w", str(wheel_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(wheel_dir.glob("portable_alpha_extension_model-*.whl"))
    assert wheels, "expected a built project wheel"
    return wheels[-1]


@pytest.mark.slow
def test_wheel_install_exposes_bundled_dashboard_samples(tmp_path: Path) -> None:
    """Non-editable installs must resolve index, asset, and portfolio samples (#2285)."""
    wheel_path = _build_wheel(tmp_path)
    install_root = tmp_path / "install-env"
    venv.EnvBuilder(with_pip=True).create(install_root)
    python = install_root / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", str(wheel_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    probe = """
from dashboard.utils import (
    bundled_asset_timeseries_path,
    bundled_portfolio_template_path,
    bundled_sample_index_path,
    load_bundled_asset_returns,
)
from pa_core.schema import load_scenario

index_path = bundled_sample_index_path()
asset_path = bundled_asset_timeseries_path()
portfolio_path = bundled_portfolio_template_path()
assert index_path.is_file(), index_path
assert asset_path.is_file(), asset_path
assert portfolio_path.is_file(), portfolio_path
load_bundled_asset_returns()
load_scenario(portfolio_path)
print("ok")
"""
    completed = subprocess.run(
        [python, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "ok" in completed.stdout


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
    grid_config = build_sample_model_config(
        analysis_mode="alpha_shares",
        external_pa_alpha_min_pct=25.0,
        external_pa_alpha_max_pct=75.0,
        external_pa_alpha_step_pct=5.0,
        active_share_min_pct=20.0,
        active_share_max_pct=100.0,
        active_share_step_pct=5.0,
    )

    assert stress_config.financing_mode == SAMPLE_FINANCING_MODE
    assert grid_config.financing_mode == SAMPLE_FINANCING_MODE
    assert stress_config.financing_mode in {"per_path", "broadcast"}
    assert grid_config.financing_mode in {"per_path", "broadcast"}
    assert grid_config.analysis_mode == "alpha_shares"
    assert grid_config.external_pa_alpha_min_pct == 25.0
    assert grid_config.external_pa_alpha_max_pct == 75.0
    assert grid_config.external_pa_alpha_step_pct == 5.0
    assert grid_config.active_share_min_pct == 20.0
    assert grid_config.active_share_max_pct == 100.0
    assert grid_config.active_share_step_pct == 5.0
