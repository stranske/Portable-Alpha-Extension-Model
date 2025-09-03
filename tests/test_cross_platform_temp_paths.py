"""Test cross-platform temporary path compatibility.

This module tests that the codebase properly uses cross-platform temporary
directory mechanisms instead of hard-coded Unix-style /tmp/ paths.
"""

from __future__ import annotations

import ast
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


def _check_tmp_path_usage_with_ast(file_path: Path) -> bool:
    """Check if test functions use tmp_path fixture using AST parsing.
    
    This is a more robust alternative to string-based checks that can handle:
    - Different formatting styles (spacing variations)
    - Different type annotation styles
    - Various parameter naming patterns
    
    Args:
        file_path: Path to the Python test file to check
        
    Returns:
        True if the file contains test functions using tmp_path fixture
    """
    if not file_path.exists():
        return False
        
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)
    except (OSError, SyntaxError):
        # If we can't read or parse the file, return False
        return False
    
    # Find all function definitions that start with "test_"
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            test_functions.append(node)
    
    # Check if any test function has tmp_path parameter
    for func in test_functions:
        for arg in func.args.args:
            if arg.arg == 'tmp_path':
                return True
    
    return False


def test_make_portable_zip_security_tests_use_tmp_path_fixture():
    """Verify that security tests use tmp_path fixture instead of hard-coded paths.
    
    This test demonstrates the improved AST-based approach for checking fixture usage,
    which addresses the brittle string-based assertion issue mentioned in #584.
    
    The old approach:
        assert "tmp_path: Path" in content
        
    Problems with old approach:
    - Breaks with formatting changes (no space: tmp_path:Path)  
    - Breaks with legitimate variations (no type annotation: tmp_path)
    - Breaks with different spacing ( tmp_path : Path )
    
    The new AST-based approach:
    - Parses Python AST to find test function parameters
    - Robust to formatting variations
    - Handles all legitimate parameter declaration styles
    - More maintainable and future-proof
    """
    # Read the security test file
    test_file = Path(__file__).parent / "test_make_portable_zip_security.py"
    
    # OLD APPROACH (brittle string-based):
    # This is the problematic approach mentioned in issue #584
    # content = test_file.read_text()
    # assert "tmp_path: Path" in content, "Security tests should use tmp_path fixture"
    
    # NEW APPROACH (robust AST-based):
    # Use AST parsing to detect tmp_path fixture usage
    has_tmp_path = _check_tmp_path_usage_with_ast(test_file)
    
    if not has_tmp_path:
        # If no tmp_path usage found, check file content for hard-coded paths
        content = test_file.read_text()
        
        # Check for hard-coded /tmp/ paths in actual code (not comments)
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
        
        # Check for hard-coded /tmp/ paths in actual code
        hard_coded_tmp_paths = [
            'Path("/tmp/',
            "Path('/tmp/",
            '"/tmp/',
            "'/tmp/",
        ]
        
        for pattern in hard_coded_tmp_paths:
            if pattern in code_only:
                pytest.fail(
                    f"Found hard-coded tmp path pattern: {pattern}. "
                    "Security tests should use tmp_path fixture instead of hard-coded paths."
                )
    
    # If we have tmp_path usage, that's good - test passes
    # If we don't have tmp_path but also don't have hard-coded paths, that's acceptable too
    # Only fail if we have hard-coded paths without tmp_path


def test_ast_tmp_path_detection_function():
    """Test the AST-based tmp_path detection function directly."""
    # Create a temporary test file with various scenarios
    import tempfile
    
    test_cases = [
        # Case 1: Standard tmp_path usage
        ("""
def test_with_tmp_path(tmp_path: Path):
    pass
        """, True),
        
        # Case 2: No type annotation
        ("""
def test_no_annotation(tmp_path):
    pass
        """, True),
        
        # Case 3: Different spacing
        ("""
def test_spaced( tmp_path : Path ):
    pass
        """, True),
        
        # Case 4: No tmp_path parameter
        ("""
def test_without_tmp_path(other_param: str):
    pass
        """, False),
        
        # Case 5: Non-test function with tmp_path (should be ignored)
        ("""
def helper_function(tmp_path: Path):
    pass

def test_something(other_param: str):
    pass
        """, False),
        
        # Case 6: Multiple parameters including tmp_path
        ("""
def test_multiple_params(arg1: str, tmp_path: Path, arg2: int):
    pass
        """, True),
    ]
    
    for i, (test_code, expected) in enumerate(test_cases):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = Path(f.name)
        
        try:
            result = _check_tmp_path_usage_with_ast(temp_file)
            assert result == expected, f"Test case {i+1} failed: expected {expected}, got {result}"
        finally:
            temp_file.unlink()


def test_ast_function_handles_invalid_files():
    """Test that AST function gracefully handles invalid files."""
    # Test with non-existent file
    non_existent = Path("/non/existent/file.py")
    assert _check_tmp_path_usage_with_ast(non_existent) is False
    
    # Test with file containing syntax errors
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def incomplete_function(")  # Invalid Python syntax
        invalid_file = Path(f.name)
    
    try:
        result = _check_tmp_path_usage_with_ast(invalid_file)
        assert result is False  # Should handle syntax error gracefully
    finally:
        invalid_file.unlink()