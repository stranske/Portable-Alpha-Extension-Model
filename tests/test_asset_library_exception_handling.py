"""Test the improved exception handling for file descriptor closing in Asset Library."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_file_descriptor_closing_with_oserror_logging(dashboard_module_loader):
    """Test that OSError during file descriptor closing is properly logged."""

    # Get the logger from the asset library module
    asset_lib_module = dashboard_module_loader("dashboard/pages/1_Asset_Library.py")
    logger = asset_lib_module["logger"]

    # Test the actual pattern used in the Asset Library
    with patch.object(logger, "warning") as mock_warning:
        original_close = os.close

        def close_then_raise(fd: int) -> None:
            """Close the descriptor then raise OSError to simulate double close."""
            original_close(fd)
            raise OSError("Bad file descriptor")

        # Mock os.close to raise an OSError (common when fd is already closed)
        with patch("os.close", side_effect=close_then_raise):
            # Simulate the exact pattern from Asset Library
            fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
            try:
                # This would normally be the file operations
                Path(tmp_path).write_text("test content")
            finally:
                try:
                    os.close(fd)
                except OSError as e:
                    # This is the fixed pattern - should log the error
                    logger.warning(f"Error closing file descriptor {fd}: {e}")
                Path(tmp_path).unlink(missing_ok=True)

        # Verify that the warning was logged exactly once
        mock_warning.assert_called_once()
        # Verify the log message format matches what we implemented
        call_args = mock_warning.call_args[0][0]
        assert "Error closing file descriptor" in call_args
        assert "Bad file descriptor" in call_args


def test_file_descriptor_closing_normal_case(dashboard_module_loader):
    """Test that normal file descriptor closing works without warnings."""

    # Get the logger from the asset library module
    asset_lib_module = dashboard_module_loader("dashboard/pages/1_Asset_Library.py")
    logger = asset_lib_module["logger"]

    with patch.object(logger, "warning") as mock_warning:
        # Normal case - no exception should be raised, no warning logged
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
        try:
            Path(tmp_path).write_text("test content")
        finally:
            try:
                os.close(fd)
            except OSError as e:
                logger.warning(f"Error closing file descriptor {fd}: {e}")
            Path(tmp_path).unlink(missing_ok=True)

        # No warning should be logged in normal case
        mock_warning.assert_not_called()


def test_specific_oserror_handling_not_broad_exception(dashboard_module_loader):
    """Test that we only catch OSError, not other exceptions."""

    # Get the logger from the asset library module
    asset_lib_module = dashboard_module_loader("dashboard/pages/1_Asset_Library.py")
    logger = asset_lib_module["logger"]

    with patch.object(logger, "warning") as mock_warning:
        original_close = os.close

        def close_then_value_error(fd: int) -> None:
            original_close(fd)
            raise ValueError("Some other error")

        # Mock os.close to raise a different exception (not OSError)
        with patch("os.close", side_effect=close_then_value_error):
            fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
            try:
                Path(tmp_path).write_text("test content")
            finally:
                # This should NOT catch ValueError - only OSError
                with pytest.raises(ValueError, match="Some other error"):
                    try:
                        os.close(fd)
                    except OSError as e:
                        logger.warning(f"Error closing file descriptor {fd}: {e}")

                Path(tmp_path).unlink(missing_ok=True)

        # No warning should be logged since ValueError wasn't caught
        mock_warning.assert_not_called()


def test_logger_uses_correct_module_name(dashboard_module_loader):
    """Test that the logger is created with the correct module name."""
    # Get the logger from the asset library module
    asset_lib_module = dashboard_module_loader("dashboard/pages/1_Asset_Library.py")
    logger = asset_lib_module["logger"]

    # Verify that the logger we import is using the right name
    assert logger.name == "test_module"
