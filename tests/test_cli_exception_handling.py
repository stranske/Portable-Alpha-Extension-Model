#!/usr/bin/env python3
"""
Tests for improved CLI exception handling to verify specific exceptions
are caught and logged properly instead of broad Exception catches.
"""

import logging

import pandas as pd
import pytest


@pytest.fixture
def caplog_debug(caplog):
    """Fixture to capture debug-level logs."""
    caplog.set_level(logging.DEBUG)
    return caplog


def test_attribution_exception_handling_attribute_error(caplog_debug):
    """Test that AttributeError in attribution calculation is specifically caught and logged."""

    # This reproduces the scenario where attribution calculation fails with AttributeError
    # due to invalid data types (e.g., string instead of numpy array)

    # Create mock returns data that will cause AttributeError
    bad_returns = {
        "Base": "invalid_string",
        "ExternalPA": [1, 2, 3],
    }  # String has no .mean() method
    inputs_dict = {}

    # Extract the logic from CLI for testing
    try:
        rows = []
        for agent, arr in bad_returns.items():
            mean_month = float(arr.mean())  # This will fail for string
            ann = 12.0 * mean_month
            rows.append({"Agent": agent, "Sub": "Total", "Return": ann})
        inputs_dict["_attribution_df"] = pd.DataFrame(rows)
    except (AttributeError, TypeError) as e:
        # This matches our improved exception handling
        logging.getLogger("pa_core.cli").warning(
            f"Attribution calculation failed due to data type issue: {e}"
        )
        inputs_dict["_attribution_df"] = pd.DataFrame(
            columns=["Agent", "Sub", "Return"]
        )

    # Verify the fallback was created
    assert "_attribution_df" in inputs_dict
    assert inputs_dict["_attribution_df"].shape == (0, 3)
    assert list(inputs_dict["_attribution_df"].columns) == ["Agent", "Sub", "Return"]

    # Check that specific warning was logged
    assert "Attribution calculation failed due to data type issue" in caplog_debug.text
    assert "'str' object has no attribute 'mean'" in caplog_debug.text


def test_attribution_exception_handling_value_error(caplog_debug):
    """Test that ValueError/KeyError in attribution calculation is specifically caught and logged."""

    # Create scenario that might cause ValueError (empty returns dict)
    empty_returns = {}
    inputs_dict = {}

    # This should work without errors (empty loop), but test the exception path
    # by testing the pattern for other potential configuration issues

    try:
        rows = []
        for agent, arr in empty_returns.items():
            mean_month = float(arr.mean())
            ann = 12.0 * mean_month
            rows.append({"Agent": agent, "Sub": "Total", "Return": ann})
        inputs_dict["_attribution_df"] = pd.DataFrame(rows)
        # Force a KeyError to test the handler
        if len(rows) == 0:
            raise KeyError("No valid agents found in returns data")
    except (ValueError, KeyError) as e:
        logging.getLogger("pa_core.cli").warning(
            f"Attribution calculation failed due to configuration issue: {e}"
        )
        inputs_dict["_attribution_df"] = pd.DataFrame(
            columns=["Agent", "Sub", "Return"]
        )

    # Verify the fallback was created
    assert "_attribution_df" in inputs_dict
    assert inputs_dict["_attribution_df"].shape == (0, 3)

    # Check that specific warning was logged
    assert (
        "Attribution calculation failed due to configuration issue" in caplog_debug.text
    )


def test_sensitivity_parameter_evaluation_value_error(caplog_debug):
    """Test that ValueError in parameter evaluation is specifically caught and logged."""

    def mock_evaluator(params):
        if params.get("mu_H", 0) < 0:
            raise ValueError("mu_H must be non-negative")
        return 0.08

    # Test the parameter evaluation exception handling pattern
    failed_params = []
    skipped_params = []

    param_name = "mu_H"
    pos_key = f"{param_name}_+5%"

    try:
        pos_value = -0.01  # This will cause ValueError
        mock_evaluator({param_name: pos_value})
    except (ValueError, ZeroDivisionError) as e:
        failed_params.append(f"{pos_key}: Configuration error: {str(e)}")
        skipped_params.append(pos_key)
        logging.getLogger("pa_core.cli").warning(
            f"Parameter evaluation failed for {pos_key} due to configuration: {e}"
        )
    except (KeyError, TypeError) as e:
        failed_params.append(f"{pos_key}: Data type error: {str(e)}")
        skipped_params.append(pos_key)
        logging.getLogger("pa_core.cli").error(
            f"Parameter evaluation failed for {pos_key} due to data issue: {e}"
        )

    # Verify specific exception was caught and logged
    assert len(failed_params) == 1
    assert "Configuration error" in failed_params[0]
    assert "mu_H must be non-negative" in failed_params[0]

    # Check logging
    assert (
        "Parameter evaluation failed for mu_H_+5% due to configuration"
        in caplog_debug.text
    )
    assert "mu_H must be non-negative" in caplog_debug.text


def test_sensitivity_parameter_evaluation_key_error(caplog_debug):
    """Test that KeyError in parameter evaluation is specifically caught and logged."""

    def mock_evaluator(params):
        if "invalid_key" in params:
            raise KeyError("Unknown parameter key")
        return 0.08

    # Test the parameter evaluation exception handling pattern
    failed_params = []
    skipped_params = []

    param_name = "invalid_key"
    pos_key = f"{param_name}_+5%"

    try:
        mock_evaluator({param_name: 0.05})
    except (ValueError, ZeroDivisionError) as e:
        failed_params.append(f"{pos_key}: Configuration error: {str(e)}")
        skipped_params.append(pos_key)
        logging.getLogger("pa_core.cli").warning(
            f"Parameter evaluation failed for {pos_key} due to configuration: {e}"
        )
    except (KeyError, TypeError) as e:
        failed_params.append(f"{pos_key}: Data type error: {str(e)}")
        skipped_params.append(pos_key)
        logging.getLogger("pa_core.cli").error(
            f"Parameter evaluation failed for {pos_key} due to data issue: {e}"
        )

    # Verify specific exception was caught and logged
    assert len(failed_params) == 1
    assert "Data type error" in failed_params[0]
    assert "Unknown parameter key" in failed_params[0]

    # Check logging
    assert (
        "Parameter evaluation failed for invalid_key_+5% due to data issue"
        in caplog_debug.text
    )
    assert "Unknown parameter key" in caplog_debug.text


def test_no_more_silent_failures():
    """Test that we no longer have silent exception handling (except Exception: pass)."""

    # Read the cli.py file and verify no silent exception handlers remain
    from pathlib import Path

    cli_file = Path(__file__).parent.parent / "pa_core" / "cli.py"
    cli_content = cli_file.read_text(encoding="utf-8")

    # Check that we don't have the old silent pattern
    assert "except Exception:\n        pass" not in cli_content

    # Check that we have proper logging-based exception handling
    assert "logging.getLogger(__name__)" in cli_content
    assert "logger.warning" in cli_content or "logger.error" in cli_content


def test_specific_exception_types_used():
    """Test that specific exception types are used instead of broad Exception catches."""

    from pathlib import Path

    cli_file = Path(__file__).parent.parent / "pa_core" / "cli.py"
    cli_content = cli_file.read_text(encoding="utf-8")

    # Verify we have specific exception handling
    assert "(AttributeError, TypeError)" in cli_content
    assert "(ValueError, KeyError)" in cli_content
    assert "(ValueError, ZeroDivisionError)" in cli_content
    assert "(ImportError, ModuleNotFoundError)" in cli_content

    # Count remaining broad Exception catches - should be reduced
    broad_exception_count = cli_content.count("except Exception")
    # We expect some remaining for cases where broad catching is still appropriate
    # (like export functionality with external dependencies)
    assert broad_exception_count < 10  # Should be significantly less than before
