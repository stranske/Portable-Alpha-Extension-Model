"""Test cross-platform temporary path compatibility.

This module tests that the codebase properly uses cross-platform temporary
directory mechanisms instead of hard-coded Unix-style /tmp/ paths.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from unittest import mock

import pytest


def test_tempfile_uses_platform_appropriate_directory():
    """Test that tempfile module uses platform-appropriate directories."""
    # Get the temporary directory
    temp_dir = tempfile.gettempdir()
    
    # On Windows, this should not be /tmp
    if sys.platform == "win32":
        assert not temp_dir.startswith("/tmp"), f"Windows should not use /tmp, got {temp_dir}"
        # Windows typically uses something like C:\Users\...\AppData\Local\Temp
        assert temp_dir.replace("\\", "/").count("/") >= 2, "Windows temp path should be nested"
    else:
        # On Unix-like systems, /tmp is common but not required
        assert temp_dir, "Temp directory should exist"
    
    # Test that NamedTemporaryFile creates files in the correct directory
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)
        assert temp_path.parent == Path(temp_dir), f"Temp file should be in {temp_dir}"


def test_security_demo_uses_cross_platform_temp(tmp_path: Path):
    """Test that the security demo uses cross-platform temporary files."""
    # Import the security demo module and test its functions directly
    import sys
    from io import StringIO
    
    # Add the scripts directory to the path temporarily
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    
    try:
        import security_demo
        
        # Capture stdout to verify the demo runs without errors
        old_stdout = sys.stdout
        captured_output = StringIO()
        
        try:
            sys.stdout = captured_output
            
            # Run just the secure implementation demo function
            security_demo.demonstrate_secure_implementation()
            
            output = captured_output.getvalue()
            
            # Should contain success message
            assert "✅ Secure execution successful!" in output
            
            # Should not contain error messages
            assert "❌" not in output
            
        finally:
            sys.stdout = old_stdout
            
    finally:
        # Clean up sys.path
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_temp_paths_work_on_different_platforms(tmp_path: Path, platform):
    """Test that our temporary path usage works correctly across platforms."""
    # Mock the platform
    with mock.patch("sys.platform", platform):
        # Test creating a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write("print('Hello World')\n")
            temp_path = Path(temp_file.name)
        
        try:
            # File should exist and be readable
            assert temp_path.exists(), f"Temp file should exist on {platform}"
            content = temp_path.read_text()
            assert content == "print('Hello World')\n"
            
            # Path should be absolute
            assert temp_path.is_absolute(), f"Temp path should be absolute on {platform}"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


def test_make_portable_zip_security_tests_use_tmp_path_fixture():
    """Verify that security tests use tmp_path fixture instead of hard-coded paths."""
    # Read the security test file
    test_file = Path(__file__).parent / "test_make_portable_zip_security.py"
    content = test_file.read_text()
    
    # Should use tmp_path fixture
    assert "tmp_path: Path" in content, "Security tests should use tmp_path fixture"
    
    # Should not have hard-coded /tmp/ paths in the actual test code
    lines = content.splitlines()
    code_lines = []
    
    for line in lines:
        # Skip comment lines and docstrings for this check
        stripped = line.strip()
        if stripped.startswith(('"""', "'''", "#")):
            continue
        if '"""' in line or "'''" in line:
            continue
            
        code_lines.append(line)
    
    code_only = '\n'.join(code_lines)
    
    # Check for hard-coded /tmp/ paths in actual code (not comments)
    hard_coded_tmp_paths = [
        'Path("/tmp/',
        "Path('/tmp/",
        '"/tmp/',
        "'/tmp/",
    ]
    
    for pattern in hard_coded_tmp_paths:
        assert pattern not in code_only, f"Found hard-coded tmp path pattern: {pattern}"
    
    # Should use tmp_path-based paths instead
    assert "tmp_path /" in code_only, "Should use tmp_path-based path construction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])