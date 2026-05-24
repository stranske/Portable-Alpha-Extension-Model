"""Tests for the LLM lockfile validation helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_validate_lockfile_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_lockfile.py"
    spec = importlib.util.spec_from_file_location("validate_lockfile", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load validate_lockfile module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_requirements_match_lockfile() -> None:
    validate_lockfile = _load_validate_lockfile_module()

    assert validate_lockfile.main() == 0


def test_workflow_drift_requirements_must_stay_strictly_pinned() -> None:
    validate_lockfile = _load_validate_lockfile_module()
    requirements = [
        validate_lockfile.Requirement("langchain>=1.3"),
        validate_lockfile.Requirement("requests~=2.34"),
    ]

    assert validate_lockfile._validate_strict_workflow_pins(requirements) == [
        "- workflow pin must use one exact == version: langchain",
        "- workflow pin must use one exact == version: requests",
    ]
