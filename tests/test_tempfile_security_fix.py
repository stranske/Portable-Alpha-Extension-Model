"""
Test that temporary file security fixes work correctly.

This test verifies that after the security fix, no temporary files are left
behind when using the functions that previously used delete=False.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pa_core.data import load_index_returns
from pa_core.pa import _convert_csv_to_yaml


def test_load_index_returns_leaves_no_temp_files(monkeypatch):
    """Test that load_index_returns doesn't leave temporary files behind."""
    # Isolate from shared system temp to avoid races with parallel xdist workers
    with tempfile.TemporaryDirectory() as isolated_temp_dir:
        monkeypatch.setenv("TMPDIR", isolated_temp_dir)
        monkeypatch.setenv("TEMP", isolated_temp_dir)
        monkeypatch.setenv("TMP", isolated_temp_dir)
        tempfile.tempdir = None  # force re-resolution of gettempdir()

        csv_path = os.path.join(isolated_temp_dir, "test_input.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Date,Monthly_TR\n")
            f.write("2020-01-01,0.01\n")
            f.write("2020-02-01,0.02\n")
            f.write("2020-03-01,0.03\n")

        series = load_index_returns(csv_path)
        assert isinstance(series, pd.Series)

        remaining_files = os.listdir(isolated_temp_dir)
        suspicious_files = [
            f
            for f in remaining_files
            if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
        ]

        assert len(suspicious_files) == 0, f"Temporary files left behind: {suspicious_files}"


def test_csv_to_yaml_conversion_leaves_no_temp_files(monkeypatch):
    """Test that CSV to YAML conversion doesn't leave temporary files behind."""
    # Isolate from shared system temp to avoid races with parallel xdist workers
    with tempfile.TemporaryDirectory() as isolated_temp_dir:
        monkeypatch.setenv("TMPDIR", isolated_temp_dir)
        monkeypatch.setenv("TEMP", isolated_temp_dir)
        monkeypatch.setenv("TMP", isolated_temp_dir)
        tempfile.tempdir = None  # force re-resolution of gettempdir()

        csv_path = os.path.join(isolated_temp_dir, "params.csv")
        yaml_path = os.path.join(isolated_temp_dir, "params.yml")

        with open(csv_path, "w", encoding="utf-8") as csv_f:
            csv_f.write("Parameter,Value\n")
            csv_f.write("Analysis mode,returns\n")
            csv_f.write("Number of simulations,100\n")

        _convert_csv_to_yaml(csv_path, yaml_path)
        assert Path(yaml_path).exists()

        remaining_files = os.listdir(isolated_temp_dir)
        suspicious_files = [
            f
            for f in remaining_files
            if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
        ]

        assert len(suspicious_files) == 0, f"Temporary files left behind: {suspicious_files}"


def test_temp_files_auto_cleanup_on_exception():
    """Test that temp files are cleaned up even when exceptions occur."""
    # Use an isolated temp directory to avoid race conditions with parallel tests
    with tempfile.TemporaryDirectory() as isolated_temp_dir:
        csv_path = os.path.join(isolated_temp_dir, "test_input.csv")

        # Create a test file
        with open(csv_path, "w") as f:
            f.write("Date,Monthly_TR\n")
            f.write("2020-01-01,invalid_data\n")  # This might cause issues

        try:
            # Even if this raises an exception, we want to verify no extra temp files
            try:
                load_index_returns(csv_path)
            except Exception:
                pass  # Ignore any exceptions - we're testing cleanup
        finally:
            # The test file should be removable (not locked)
            if os.path.exists(csv_path):
                os.remove(csv_path)

        # Check that no unexpected temp files were created in our isolated directory
        remaining_files = os.listdir(isolated_temp_dir)
        suspicious_files = [
            f
            for f in remaining_files
            if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
        ]

        assert (
            len(suspicious_files) == 0
        ), f"Temporary files left behind after exception: {suspicious_files}"


def test_no_delete_false_in_codebase():
    """Test that delete=False has been removed from all relevant files."""
    # Check that we haven't missed any delete=False usage in key files
    monitored_files = [
        "dashboard/pages/1_Asset_Library.py",
        "dashboard/pages/2_Portfolio_Builder.py",
        "dashboard/pages/3_Scenario_Wizard.py",
    ]

    for file_path in monitored_files:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            # Should not find delete=False anymore
            assert "delete=False" not in content, f"delete=False still found in {file_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
