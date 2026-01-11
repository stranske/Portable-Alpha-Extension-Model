"""Tests for facade export function."""

from pathlib import Path

import pandas as pd
import pytest

from pa_core.config import load_config
from pa_core.facade import ExportOptions, RunOptions, export, run_single, run_sweep


@pytest.fixture
def run_artifacts():
    """Create minimal run artifacts for testing."""
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    return run_single(cfg, idx, RunOptions(seed=123))


@pytest.fixture
def sweep_artifacts():
    """Create minimal sweep artifacts for testing."""
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    sweep_params = {
        "analysis_mode": "vol_mult",
        "sd_multiple_min": 1.0,
        "sd_multiple_max": 1.0,
        "sd_multiple_step": 1.0,
    }
    return run_sweep(cfg, idx, sweep_params, RunOptions(seed=123))


def test_export_run_artifacts(run_artifacts, tmp_path: Path) -> None:
    """Test exporting RunArtifacts to Excel."""
    output_file = tmp_path / "run_output.xlsx"

    result = export(run_artifacts, output_file)

    assert result == output_file
    assert output_file.exists()
    # Verify Excel file has expected sheets
    xl = pd.ExcelFile(output_file)
    assert "Summary" in xl.sheet_names
    assert "Inputs" in xl.sheet_names


def test_export_sweep_artifacts(sweep_artifacts, tmp_path: Path) -> None:
    """Test exporting SweepArtifacts to Excel."""
    output_file = tmp_path / "sweep_output.xlsx"

    result = export(sweep_artifacts, output_file)

    assert result == output_file
    assert output_file.exists()


def test_export_with_options(run_artifacts, tmp_path: Path) -> None:
    """Test exporting with ExportOptions."""
    output_file = tmp_path / "output_with_options.xlsx"
    options = ExportOptions(pivot=True, include_charts=False)

    result = export(run_artifacts, output_file, options)

    assert result == output_file
    assert output_file.exists()


def test_export_with_visualizations(run_artifacts, tmp_path: Path) -> None:
    """Test exporting visualization HTML files alongside Excel output."""
    output_file = tmp_path / "output_with_viz.xlsx"
    viz_dir = tmp_path / "viz"
    options = ExportOptions(include_visualizations=True, viz_output_dir=viz_dir)

    result = export(run_artifacts, output_file, options)

    assert result == output_file
    assert output_file.exists()
    for name in ("risk_return", "cvar_return", "return_distribution", "risk_metrics"):
        assert (viz_dir / f"{name}.html").exists()


def test_export_invalid_artifacts(tmp_path: Path) -> None:
    """Test that export raises ValueError for invalid artifacts type."""
    output_file = tmp_path / "invalid.xlsx"

    with pytest.raises(ValueError, match="Unsupported artifacts type"):
        export("not_artifacts", output_file)  # type: ignore[arg-type]
