"""Ensure the lock file captures every dependency declared in pyproject.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict

import tomllib

_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<", "===")


def _split_spec(raw: str) -> tuple[str, str | None]:
    """Split a dependency spec into (package_name, condition_marker)."""
    entry = raw.strip().strip(",").strip('"')
    # Handle conditional markers (e.g., ; python_version < "3.11")
    condition = None
    if ";" in entry:
        entry, condition = entry.split(";", 1)
        entry = entry.strip()
        condition = condition.strip()
    
    for operator in _OPERATORS:
        if operator in entry:
            name, _ = entry.split(operator, 1)
            return name.strip().split("[")[0], condition
    return entry.strip().split("[")[0], condition


def _is_applicable_condition(condition: str | None, current_python: tuple[int, int]) -> bool:
    """Check if a conditional marker applies to the current Python version."""
    if condition is None:
        return True
    
    # Handle python_version conditions - be flexible with quote matching
    match = re.search(r'python_version\s*([<>=!]+)\s*["\']?(\d+\.\d+)', condition)
    if match:
        operator, version_str = match.groups()
        required_version = tuple(map(int, version_str.split(".")))
        
        # For < operator, if current Python is >= the version, this dep won't be in lock
        if operator in ("<", "<="):
            # tomli; python_version < "3.11" means it's only for < 3.11
            # If we're on 3.12, skip it
            if current_python >= required_version:
                return False
        # For > or >= operator, if current Python is < the version, skip
        elif operator in (">", ">="):
            if current_python < required_version:
                return False
    
    return True


def _load_lock_versions(path: Path) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            continue
        # Handle conditional markers in lock file (e.g., ; sys_platform == 'win32')
        if "==" in stripped:
            entry = stripped.split(";")[0].strip() if ";" in stripped else stripped
            if "==" in entry:
                name, version = entry.split("==", 1)
                versions[name.lower().strip()] = version.strip()
    return versions


def test_all_pyproject_dependencies_are_in_lock() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    current_python = sys.version_info[:2]

    declared = set()
    for entry in project.get("dependencies", []):
        pkg_name, condition = _split_spec(entry)
        pkg_name = pkg_name.lower()
        # Skip self-references
        if pkg_name == pyproject.get("project", {}).get("name", "").lower().replace("_", "-"):
            continue
        # Skip conditional dependencies that don't apply to current Python
        if not _is_applicable_condition(condition, current_python):
            continue
        declared.add(pkg_name)

    for group in project.get("optional-dependencies", {}).values():
        for entry in group:
            pkg_name, condition = _split_spec(entry)
            pkg_name = pkg_name.lower()
            # Skip self-references
            if pkg_name == pyproject.get("project", {}).get("name", "").lower().replace("_", "-"):
                continue
            # Skip conditional dependencies that don't apply to current Python
            if not _is_applicable_condition(condition, current_python):
                continue
            declared.add(pkg_name)

    lock_versions = _load_lock_versions(Path("requirements.lock"))

    missing = []
    for dependency in sorted(declared):
        normalised = dependency.replace("-", "_")
        # Check both forms: package-name and package_name
        if dependency not in lock_versions and normalised not in lock_versions:
            missing.append(dependency)

    assert not missing, "requirements.lock is missing pinned versions for: " + ", ".join(missing)
