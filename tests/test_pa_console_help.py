from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _console_help(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    # Exercise the installed console script against this checkout, even when
    # the interpreter belongs to another linked worktree's virtual environment.
    console = Path(sys.executable).with_name("pa")
    return subprocess.run(
        [str(console), *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("help_args", [("--help",), ("-h",), ()])
def test_pa_run_help_lists_core_flags(tmp_path: Path, help_args: tuple[str, ...]) -> None:
    result = _console_help(tmp_path, "run", *help_args)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--index" in result.stdout
    assert "--png" in result.stdout
    assert "deprecated" not in result.stderr.lower()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("help_args", [("--help",), ("-h",), ()])
def test_pa_validate_help_lists_schema_and_path(tmp_path: Path, help_args: tuple[str, ...]) -> None:
    result = _console_help(tmp_path, "validate", *help_args)
    assert result.returncode == 0, result.stderr
    assert "--schema" in result.stdout
    assert "path" in result.stdout
    assert not list(tmp_path.iterdir())


def test_pa_top_level_help_preserves_subcommands(tmp_path: Path) -> None:
    result = _console_help(tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    for command in ("run", "validate", "sweep", "calibrate", "convert"):
        assert command in result.stdout
