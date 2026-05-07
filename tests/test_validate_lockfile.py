"""Tests for the LLM lockfile validation helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_validate_lockfile_module():
    module_path = Path("scripts/validate_lockfile.py")
    spec = importlib.util.spec_from_file_location("validate_lockfile", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llm_requirements_match_lockfile() -> None:
    validate_lockfile = _load_validate_lockfile_module()

    assert validate_lockfile.main() == 0
