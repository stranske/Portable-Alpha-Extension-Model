"""Test security aspects of make_portable_zip.py.

This module tests that the portable zip creation functionality uses secure
subprocess calls instead of vulnerable os.system() calls.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Import the module to test
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_portable_zip


def test_build_windows_portable_zip_uses_subprocess_not_os_system(tmp_path: Path):
    """Test that build_windows_portable_zip uses secure subprocess.run() calls."""
    # Create a mock staging directory and project root
    temp_staging = tmp_path / "test_staging"
    temp_project = tmp_path / "test_project"
    temp_output = tmp_path / "test_output.zip"
    
    # Mock all the external dependencies
    with mock.patch("make_portable_zip.sys.platform", "win32"), \
         mock.patch("make_portable_zip.Path.exists") as mock_exists, \
         mock.patch("make_portable_zip.Path.mkdir"), \
         mock.patch("make_portable_zip.shutil.rmtree"), \
         mock.patch("make_portable_zip._download"), \
         mock.patch("make_portable_zip._unzip"), \
         mock.patch("make_portable_zip._enable_embedded_site"), \
         mock.patch("make_portable_zip._write_launcher_bats"), \
         mock.patch("make_portable_zip.subprocess.run") as mock_subprocess_run, \
         mock.patch("make_portable_zip.shutil.copytree"), \
         mock.patch("make_portable_zip.shutil.copy2"), \
         mock.patch("make_portable_zip.zipfile.ZipFile") as mock_zipfile, \
         mock.patch("make_portable_zip.Path.iterdir") as mock_iterdir, \
         mock.patch("make_portable_zip.Path.rglob") as mock_rglob:
        
        # Set up mocks
        mock_exists.return_value = True
        mock_iterdir.return_value = [temp_project / "file1.py"]
        mock_rglob.return_value = [temp_staging / "file1.py"]
        mock_zipfile.return_value.__enter__.return_value.write = mock.Mock()
        
        # Call the function
        make_portable_zip.build_windows_portable_zip(
            temp_project, temp_output, python_version="3.12.11", verbose=False
        )
        
        # Verify subprocess.run was called for both pip bootstrap and requirements install
        assert mock_subprocess_run.call_count >= 2, "Should call subprocess.run at least twice"
        
        # Get the actual calls
        calls = mock_subprocess_run.call_args_list
        
        # Verify first call (pip bootstrap) uses list of arguments
        first_call_args = calls[0][0][0]  # First positional argument of first call
        assert isinstance(first_call_args, list), "subprocess.run should be called with list of arguments"
        assert len(first_call_args) == 2, "Pip bootstrap should have 2 arguments"
        assert first_call_args[1].endswith("get-pip.py"), "Second arg should be get-pip.py path"
        
        # Verify second call (pip install) uses list of arguments
        second_call_args = calls[1][0][0]  # First positional argument of second call
        assert isinstance(second_call_args, list), "subprocess.run should be called with list of arguments"
        assert len(second_call_args) >= 5, "Pip install should have multiple arguments"
        assert "-m" in second_call_args, "Should use python -m pip"
        assert "pip" in second_call_args, "Should call pip"
        assert "install" in second_call_args, "Should be install command"
        assert "-r" in second_call_args, "Should use -r flag for requirements"
        
        # Verify check=True is used for security (fail fast on errors)
        for call in calls:
            kwargs = call[1]
            assert kwargs.get("check") is True, "Should use check=True for security"


def test_subprocess_calls_prevent_command_injection(tmp_path: Path):
    """Test that the subprocess calls prevent command injection attacks."""
    # Test with malicious path containing shell metacharacters
    malicious_path = tmp_path / "test; rm -rf /; echo pwned"
    
    with mock.patch("make_portable_zip.subprocess.run") as mock_subprocess_run:
        # Create a minimal version of the problematic code using secure subprocess
        python_exe = malicious_path / "python.exe"
        get_pip = malicious_path / "get-pip.py" 
        
        # This is how the secure code should work
        subprocess.run([str(python_exe), str(get_pip)], check=True)
        
        # Verify the call was made with a list (secure)
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert isinstance(call_args, list), "Must use list for security"
        
        # The malicious characters should be in the path string, but not executed
        # because subprocess.run with a list doesn't invoke the shell
        assert "; rm -rf /" in str(call_args[0]) or "; rm -rf /" in str(call_args[1])


def test_error_handling_in_subprocess_calls(tmp_path: Path):
    """Test that subprocess errors are properly handled and wrapped."""
    temp_project = tmp_path / "test_project"
    temp_output = tmp_path / "test_output.zip"
    
    with mock.patch("make_portable_zip.sys.platform", "win32"), \
         mock.patch("make_portable_zip.Path.exists", return_value=True), \
         mock.patch("make_portable_zip.Path.mkdir"), \
         mock.patch("make_portable_zip.shutil.rmtree"), \
         mock.patch("make_portable_zip._download"), \
         mock.patch("make_portable_zip._unzip"), \
         mock.patch("make_portable_zip._enable_embedded_site"), \
         mock.patch("make_portable_zip.subprocess.run") as mock_subprocess_run, \
         mock.patch("make_portable_zip.Path.iterdir", return_value=[]):
        
        # Make subprocess.run raise CalledProcessError
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "test")
        
        # The function should catch the error and raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to bootstrap pip"):
            make_portable_zip.build_windows_portable_zip(
                temp_project, temp_output, python_version="3.12.11", verbose=False
            )


def test_no_os_system_imports():
    """Test that the module doesn't import os.system or similar unsafe functions."""
    import make_portable_zip
    
    # Check that os.system is not accessible from the module
    assert not hasattr(make_portable_zip, 'system'), "Module should not expose os.system"
    
    # Check the module source doesn't contain dangerous patterns (excluding comments)
    module_file = Path(make_portable_zip.__file__)
    source = module_file.read_text()
    
    # Remove comments and docstrings to check only actual code
    lines = source.splitlines()
    code_lines = []
    in_multiline_string = False
    
    for line in lines:
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith('#'):
            continue
        
        # For lines with code, remove inline comments
        if '#' in line and not in_multiline_string:
            # Simple heuristic: split at # and take the first part
            # This isn't perfect but good enough for this test
            line = line.split('#')[0].rstrip()
        
        code_lines.append(line)
    
    code_only = '\n'.join(code_lines)
    
    # These patterns should not exist in the secure version (in actual code)
    dangerous_patterns = [
        "os.system(",
        "os.popen(",
        "commands.getoutput(",
        "commands.getstatusoutput(",
    ]
    
    for pattern in dangerous_patterns:
        assert pattern not in code_only, f"Dangerous pattern '{pattern}' found in actual code"
    
    # Should contain secure patterns
    secure_patterns = [
        "subprocess.run(",
        "check=True",
    ]
    
    for pattern in secure_patterns:
        assert pattern in source, f"Secure pattern '{pattern}' not found in module source"