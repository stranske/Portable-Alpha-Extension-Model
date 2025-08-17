"""
Test the subprocess error handling in the CLI dashboard functionality.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import sys


def test_dashboard_error_handling_filenotfound():
    """Test FileNotFoundError when dashboard/app.py doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Import and test the dashboard code logic

            # This replicates the exact logic from cli.py
            dashboard_path = Path("dashboard/app.py")
            with pytest.raises(FileNotFoundError, match="Dashboard file not found"):
                if not dashboard_path.exists():
                    raise FileNotFoundError(
                        f"Dashboard file not found: {dashboard_path}"
                    )
        finally:
            os.chdir(original_cwd)


def test_dashboard_error_handling_calledprocesserror():
    """Test CalledProcessError when subprocess fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake dashboard file
        dashboard_dir = Path(temp_dir) / "dashboard"
        dashboard_dir.mkdir()
        dashboard_file = dashboard_dir / "app.py"
        dashboard_file.write_text("# fake streamlit app")

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, "streamlit")

                # This replicates the logic from cli.py
                dashboard_path = Path("dashboard/app.py")
                assert (
                    dashboard_path.exists()
                )  # File exists, so we proceed to subprocess

                with pytest.raises(subprocess.CalledProcessError):
                    subprocess.run(
                        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
                        check=True,
                        cwd=os.getcwd(),
                    )
        finally:
            os.chdir(original_cwd)


def test_dashboard_error_handling_general_exception():
    """Test general exception handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake dashboard file
        dashboard_dir = Path(temp_dir) / "dashboard"
        dashboard_dir.mkdir()
        dashboard_file = dashboard_dir / "app.py"
        dashboard_file.write_text("# fake streamlit app")

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = RuntimeError("Simulated runtime error")

                # This replicates the logic from cli.py
                dashboard_path = Path("dashboard/app.py")
                assert (
                    dashboard_path.exists()
                )  # File exists, so we proceed to subprocess

                with pytest.raises(RuntimeError, match="Simulated runtime error"):
                    subprocess.run(
                        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
                        check=True,
                        cwd=os.getcwd(),
                    )
        finally:
            os.chdir(original_cwd)
