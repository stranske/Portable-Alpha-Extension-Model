#!/usr/bin/env python3
"""Quick environment checks for agent contributors."""
from __future__ import annotations

import sys
from pathlib import Path

_MIN_PYTHON = (3, 11)


def _fmt_version(version: tuple[int, int, int]) -> str:
    return f"{version[0]}.{version[1]}.{version[2]}"


def _check_python() -> list[str]:
    errors: list[str] = []
    if sys.version_info < _MIN_PYTHON:
        errors.append(
            "Python >= 3.11 is required. " f"Detected {_fmt_version(sys.version_info[:3])}."
        )
    return errors


def _check_repo_root() -> list[str]:
    errors: list[str] = []
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "pyproject.toml").exists():
        errors.append("Run this script from the repository root (pyproject.toml not found).")
    return errors


def _check_venv() -> list[str]:
    warnings: list[str] = []
    if sys.prefix == sys.base_prefix:
        warnings.append("Virtual environment not detected. Activate .venv if available.")
    return warnings


def _check_imports() -> list[str]:
    warnings: list[str] = []
    try:
        __import__("pa_core")
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.append(f"pa_core is not importable ({exc!s}). Install dev deps.")
    return warnings


def main() -> int:
    error_checks = [
        ("Python", _check_python),
        ("Repo", _check_repo_root),
    ]
    warn_checks = [
        ("Venv", _check_venv),
        ("Imports", _check_imports),
    ]
    failures = 0
    for label, check in error_checks:
        errors = check()
        if errors:
            failures += 1
            print(f"{label}: issues found")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{label}: ok")

    for label, check in warn_checks:
        warnings = check()
        if warnings:
            print(f"{label}: warnings")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print(f"{label}: ok")

    print("\nNext steps:")
    print("  - python -m pytest tests/test_agents.py -v")
    print("  - ./dev.sh ci")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
