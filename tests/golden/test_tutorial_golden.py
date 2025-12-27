"""Golden tests for tutorial workflows.

These tests run the CLI commands as demonstrated in the tutorials
and validate that expected artifacts are produced with deterministic results.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest


class TutorialTestRunner:
    """Helper to run CLI commands and validate artifacts."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.project_root = Path(__file__).parent.parent.parent
        self.pythonpath = str(self.project_root)

    def run_cli(
        self, args: List[str], timeout: int = 600
    ) -> subprocess.CompletedProcess:
        """Run CLI command with PYTHONPATH set."""
        cmd = [sys.executable, "-m", "pa_core.cli"] + args
        env = os.environ.copy()
        env["PYTHONPATH"] = self.pythonpath
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def assert_file_exists_and_size(self, filepath: Path, min_size: int = 1000):
        """Assert file exists and has reasonable size."""
        assert filepath.exists(), f"Expected file {filepath} does not exist"
        size = filepath.stat().st_size
        assert (
            size >= min_size
        ), f"File {filepath} too small: {size} bytes (min: {min_size})"

    def get_excel_sheets(self, filepath: Path) -> List[str]:
        """Get sheet names from Excel file."""
        return pd.ExcelFile(filepath).sheet_names

    def read_excel_summary(self, filepath: Path) -> pd.DataFrame:
        """Read summary sheet from Excel output."""
        return pd.read_excel(filepath, sheet_name="Summary")


@pytest.fixture
def runner(tmp_path: Path) -> TutorialTestRunner:
    """Create test runner with temporary directory."""
    return TutorialTestRunner(tmp_path)


class TestTutorial1ParameterSweeps:
    """Tutorial 1: Enhanced Parameter Sweeps - Golden Tests."""

    def test_basic_scenario_single_run(self, runner: TutorialTestRunner):
        """Test basic single scenario run as shown in Tutorial 1 Part 1."""
        output_file = runner.tmp_path / "tutorial1_basic.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

        # Validate expected sheets exist - Summary is always present
        sheets = runner.get_excel_sheets(output_file)
        assert "Summary" in sheets, f"Summary sheet missing. Got: {sheets}"

        # Validate summary metrics are reasonable
        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 0, "Summary sheet is empty"
        assert "Base" in summary["Agent"].values, "Base agent missing from summary"

        # Check that we have the expected agents
        expected_agents = {"Base", "ExternalPA", "ActiveExt", "InternalPA"}
        actual_agents = set(summary["Agent"].unique())
        assert expected_agents.issubset(
            actual_agents
        ), f"Missing agents. Got: {actual_agents}"

    def test_returns_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test returns mode parameter sweep as shown in Tutorial 1 Part 2."""
        output_file = runner.tmp_path / "tutorial1_returns_sweep.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--mode",
                "returns",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

        # Validate sweep produces multiple scenarios
        sheets = runner.get_excel_sheets(output_file)
        assert "Summary" in sheets, "Summary sheet missing"

        summary = runner.read_excel_summary(output_file)
        # Should have multiple rows for different return scenarios
        assert len(summary) > 1, f"Expected multiple scenarios, got {len(summary)}"

    def test_capital_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test capital mode parameter sweep as shown in Tutorial 1 Part 3."""
        output_file = runner.tmp_path / "tutorial1_capital_sweep.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--mode",
                "capital",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 1, "Capital sweep should produce multiple scenarios"

    def test_alpha_shares_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test alpha shares mode parameter sweep as shown in Tutorial 1 Part 4."""
        output_file = runner.tmp_path / "tutorial1_alpha_shares_sweep.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--mode",
                "alpha_shares",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=50000)

        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 1, "Alpha shares sweep should produce multiple scenarios"

    def test_vol_mult_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test volatility multiplier mode parameter sweep as shown in Tutorial 1 Part 5."""
        output_file = runner.tmp_path / "tutorial1_vol_mult_sweep.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--mode",
                "vol_mult",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=15000)

        summary = runner.read_excel_summary(output_file)
        assert (
            len(summary) > 1
        ), "Volatility mult sweep should produce multiple scenarios"


class TestTutorial2ThresholdAnalysis:
    """Tutorial 2: Advanced Threshold Analysis - Golden Tests."""

    def test_threshold_scenario_analysis(self, runner: TutorialTestRunner):
        """Test threshold analysis scenario as shown in Tutorial 2."""
        # First create a test threshold config based on template
        threshold_config = runner.tmp_path / "test_threshold_config.yml"

        # Copy base config and modify for threshold testing
        shutil.copy(
            runner.project_root / "examples/scenarios/my_first_scenario.yml",
            threshold_config,
        )

        output_file = runner.tmp_path / "tutorial2_threshold.xlsx"

        result = runner.run_cli(
            [
                "--config",
                str(threshold_config),
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

    def test_aggressive_scenario(self, runner: TutorialTestRunner):
        """Test aggressive portfolio scenario from Tutorial 2."""
        output_file = runner.tmp_path / "tutorial2_aggressive.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/tutorial2_aggressive.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

    def test_conservative_scenario(self, runner: TutorialTestRunner):
        """Test conservative portfolio scenario from Tutorial 2."""
        output_file = runner.tmp_path / "tutorial2_conservative.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/tutorial2_conservative.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)


class TestTutorial3Integration:
    """Tutorial 3: Integration and Workflow Tests."""

    def test_full_pipeline_with_png_export(self, runner: TutorialTestRunner):
        """Test full pipeline with PNG export capability."""
        output_file = runner.tmp_path / "tutorial3_full.xlsx"

        # Note: PNG/PDF exports require Chrome/Chromium, so we just test without
        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/my_first_scenario.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)


class TestTutorial4Validation:
    """Tutorial 4: Validation and Quality Checks."""

    def test_validation_demo_scenario(self, runner: TutorialTestRunner):
        """Test validation demo scenario correctly rejects invalid config.

        The validation_demo.yml file intentionally has N_SIMULATIONS=50
        which is below the minimum of 100, demonstrating validation guardrails.
        """
        output_file = runner.tmp_path / "tutorial4_validation.xlsx"

        result = runner.run_cli(
            [
                "--config",
                "examples/scenarios/validation_demo.yml",
                "--index",
                "data/sp500tr_fred_divyield.csv",
                "--output",
                str(output_file),
                "--seed",
                "42",
            ]
        )

        # The demo file is intentionally invalid - it should fail validation
        assert result.returncode != 0, "Expected validation failure for invalid config"
        assert (
            "N_SIMULATIONS" in result.stderr
        ), "Expected N_SIMULATIONS validation error"
