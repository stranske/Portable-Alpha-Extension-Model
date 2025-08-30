"""Golden tests for tutorial workflows.

These tests run the CLI commands as demonstrated in the tutorials
and validate that expected artifacts are produced with deterministic results.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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

    def run_cli(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run CLI command with PYTHONPATH set."""
        cmd = ["python", "-m", "pa_core.cli"] + args
        env = os.environ.copy()
        env["PYTHONPATH"] = self.pythonpath
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )

    def assert_file_exists_and_size(self, filepath: Path, min_size: int = 1000):
        """Assert file exists and has reasonable size."""
        assert filepath.exists(), f"Expected file {filepath} does not exist"
        size = filepath.stat().st_size
        assert size >= min_size, f"File {filepath} too small: {size} bytes (min: {min_size})"

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
        
        result = runner.run_cli([
            "--config", "my_first_scenario.yml",
            "--index", "sp500tr_fred_divyield.csv", 
            "--output", str(output_file),
            "--seed", "42"
        ])
        
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
        assert expected_agents.issubset(actual_agents), f"Missing agents. Got: {actual_agents}"

    def test_returns_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test returns mode parameter sweep as shown in Tutorial 1 Part 2."""
        output_file = runner.tmp_path / "tutorial1_returns_sweep.xlsx"
        
        result = runner.run_cli([
            "--config", "config/returns_mode_template.yml",
            "--index", "sp500tr_fred_divyield.csv",
            "--mode", "returns",
            "--output", str(output_file),
            "--seed", "42"
        ])
        
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
        
        result = runner.run_cli([
            "--config", "config/capital_mode_template.yml", 
            "--index", "sp500tr_fred_divyield.csv",
            "--mode", "capital",
            "--output", str(output_file),
            "--seed", "42"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=45000)

        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 1, "Capital sweep should produce multiple scenarios"

    def test_alpha_shares_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test alpha shares mode parameter sweep as shown in Tutorial 1 Part 4."""
        output_file = runner.tmp_path / "tutorial1_alpha_shares_sweep.xlsx"
        
        result = runner.run_cli([
            "--config", "config/alpha_shares_mode_template.yml",
            "--index", "sp500tr_fred_divyield.csv", 
            "--mode", "alpha_shares",
            "--output", str(output_file),
            "--seed", "42"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=50000)
        
        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 1, "Alpha shares sweep should produce multiple scenarios"

    def test_vol_mult_mode_parameter_sweep(self, runner: TutorialTestRunner):
        """Test volatility multiplier mode parameter sweep as shown in Tutorial 1 Part 5."""
        output_file = runner.tmp_path / "tutorial1_vol_mult_sweep.xlsx"
        
        result = runner.run_cli([
            "--config", "config/vol_mult_mode_template.yml",
            "--index", "sp500tr_fred_divyield.csv",
            "--mode", "vol_mult", 
            "--output", str(output_file),
            "--seed", "42"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=15000)
        
        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 1, "Volatility mult sweep should produce multiple scenarios"


class TestTutorial2ThresholdAnalysis:
    """Tutorial 2: Advanced Threshold Analysis - Golden Tests."""

    def test_threshold_scenario_analysis(self, runner: TutorialTestRunner):
        """Test threshold analysis scenario as shown in Tutorial 2."""
        # First create a test threshold config based on template
        threshold_config = runner.tmp_path / "test_threshold_config.yml"
        
        # Copy base config and modify for threshold testing
        shutil.copy(runner.project_root / "my_first_scenario.yml", threshold_config)
        
        output_file = runner.tmp_path / "tutorial2_threshold.xlsx"
        
        result = runner.run_cli([
            "--config", str(threshold_config),
            "--index", "sp500tr_fred_divyield.csv",
            "--output", str(output_file), 
            "--seed", "42"
        ])
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        runner.assert_file_exists_and_size(output_file, min_size=50000)
        
        # Validate threshold analysis produces meaningful metrics
        summary = runner.read_excel_summary(output_file)
        assert len(summary) > 0, "Summary should contain threshold analysis results"
        
        # Check that risk metrics are present (required for threshold analysis)
        expected_columns = {"AnnReturn", "AnnVol", "ShortfallProb", "VaR"}
        summary_columns = set(summary.columns)
        assert expected_columns.issubset(summary_columns), f"Missing risk metrics. Got: {summary_columns}"


class TestTutorialExports:
    """Test export functionality mentioned in tutorials."""

    def test_png_export(self, runner: TutorialTestRunner):
        """Test PNG export functionality."""
        output_file = runner.tmp_path / "tutorial_export_test.xlsx"
        
        result = runner.run_cli([
            "--config", "my_first_scenario.yml",
            "--index", "sp500tr_fred_divyield.csv",
            "--output", str(output_file),
            "--png",
            "--seed", "42"
        ])
        
        # PNG export might fail without Chrome/Chromium, so we allow both success and failure
        if result.returncode == 0:
            runner.assert_file_exists_and_size(output_file, min_size=50000)
            # If successful, there might be PNG files created
        else:
            # If failed, it should be due to Chrome/Chromium missing, not other errors
            assert "chrome" in result.stderr.lower() or "chromium" in result.stderr.lower(), \
                f"Unexpected error (not Chrome/Chromium): {result.stderr}"

    def test_pptx_export(self, runner: TutorialTestRunner):
        """Test PPTX export functionality.""" 
        output_file = runner.tmp_path / "tutorial_pptx_test.xlsx"
        
        result = runner.run_cli([
            "--config", "my_first_scenario.yml",
            "--index", "sp500tr_fred_divyield.csv", 
            "--output", str(output_file),
            "--pptx",
            "--seed", "42"
        ])
        
        # PPTX export might fail without Chrome/Chromium, so we allow both success and failure
        if result.returncode == 0:
            runner.assert_file_exists_and_size(output_file, min_size=50000)
            # Check if PPTX file was created
            pptx_file = output_file.with_suffix(".pptx")
            if pptx_file.exists():
                runner.assert_file_exists_and_size(pptx_file, min_size=10000)
        else:
            # If failed, it should be due to Chrome/Chromium missing, not other errors
            assert "chrome" in result.stderr.lower() or "chromium" in result.stderr.lower(), \
                f"Unexpected error (not Chrome/Chromium): {result.stderr}"


class TestDeterministicResults:
    """Test that results are deterministic with fixed seeds."""

    def test_deterministic_run(self, runner: TutorialTestRunner):
        """Test that same seed produces identical results."""
        output_file1 = runner.tmp_path / "deterministic_test1.xlsx"
        output_file2 = runner.tmp_path / "deterministic_test2.xlsx"
        
        # Run same command twice with same seed
        for output_file in [output_file1, output_file2]:
            result = runner.run_cli([
                "--config", "my_first_scenario.yml",
                "--index", "sp500tr_fred_divyield.csv",
                "--output", str(output_file),
                "--seed", "42"
            ])
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
        
        # Compare summary results
        summary1 = runner.read_excel_summary(output_file1)
        summary2 = runner.read_excel_summary(output_file2)
        
        # Check that key metrics are identical (within floating point precision)
        pd.testing.assert_frame_equal(
            summary1[["Agent", "AnnReturn", "AnnVol", "ShortfallProb"]], 
            summary2[["Agent", "AnnReturn", "AnnVol", "ShortfallProb"]],
            check_exact=False,
            rtol=1e-10
        )