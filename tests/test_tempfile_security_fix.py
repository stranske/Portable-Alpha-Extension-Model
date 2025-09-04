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


def test_load_index_returns_leaves_no_temp_files():
    """Test that load_index_returns doesn't leave temporary files behind."""
    # Get the temp directory before the test
    temp_dir = tempfile.gettempdir()
    temp_files_before = set(os.listdir(temp_dir))

    # Create a simple CSV file and test loading
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        f.write("Date,Return\n")
        f.write("2020-01-01,0.01\n")
        f.write("2020-02-01,0.02\n")
        f.write("2020-03-01,0.03\n")
        f.flush()

        # This function should not create any persistent temp files
        series = load_index_returns(f.name)
        assert isinstance(series, pd.Series)

    # Check that no new temporary files were created
    temp_files_after = set(os.listdir(temp_dir))
    new_files = temp_files_after - temp_files_before

    # Filter out files that might be created by other processes
    # We only care about files that look like our temp files
    suspicious_files = [
        f
        for f in new_files
        if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
    ]

    assert (
        len(suspicious_files) == 0
    ), f"Temporary files left behind: {suspicious_files}"


def test_csv_to_yaml_conversion_leaves_no_temp_files():
    """Test that CSV to YAML conversion doesn't leave temporary files behind."""
    temp_dir = tempfile.gettempdir()
    temp_files_before = set(os.listdir(temp_dir))

    # Test the CSV to YAML conversion function
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as csv_f:
        csv_f.write("Parameter,Value\n")
        csv_f.write("Analysis mode,returns\n")
        csv_f.write("Number of simulations,100\n")
        csv_f.flush()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml") as yaml_f:
            # This should not create any persistent temp files
            _convert_csv_to_yaml(csv_f.name, yaml_f.name)
            assert Path(yaml_f.name).exists()

    # Check that no new temporary files were created
    temp_files_after = set(os.listdir(temp_dir))
    new_files = temp_files_after - temp_files_before

    # Filter out files that might be created by other processes
    suspicious_files = [
        f
        for f in new_files
        if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
    ]

    assert (
        len(suspicious_files) == 0
    ), f"Temporary files left behind: {suspicious_files}"


def test_temp_files_auto_cleanup_on_exception():
    """Test that temp files are cleaned up even when exceptions occur."""
    temp_dir = tempfile.gettempdir()
    temp_files_before = set(os.listdir(temp_dir))

    # Test that files are cleaned up even if an exception occurs
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
            f.write("Date,Return\n")
            f.write("2020-01-01,invalid_data\n")  # This might cause issues
            f.flush()

            # Even if this raises an exception, the temp file should be cleaned up
            try:
                load_index_returns(f.name)
            except Exception:
                pass  # Ignore any exceptions for this test
    except Exception:
        pass  # Ignore any exceptions for this test

    # Check that no new temporary files were created
    temp_files_after = set(os.listdir(temp_dir))
    new_files = temp_files_after - temp_files_before

    suspicious_files = [
        f
        for f in new_files
        if f.startswith("tmp") and (".csv" in f or ".yml" in f or ".yaml" in f)
    ]

    assert (
        len(suspicious_files) == 0
    ), f"Temporary files left behind after exception: {suspicious_files}"


def test_no_delete_false_in_codebase():
    """Test that delete=False has been removed from all relevant files."""
    # Check that we haven't missed any delete=False usage in key files
    dashboard_files = [
        "dashboard/pages/1_Asset_Library.py",
        "dashboard/pages/2_Portfolio_Builder.py",
        "dashboard/pages/3_Scenario_Wizard.py",
    ]

    test_files = [
        "tests/test_data_loading_validation.py",
        "tests/test_field_mappings.py",
        "tests/test_num_val_fix.py",
    ]

    for file_path in dashboard_files + test_files:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            content = full_path.read_text()
            # Should not find delete=False anymore
            assert (
                "delete=False" not in content
            ), f"delete=False still found in {file_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
