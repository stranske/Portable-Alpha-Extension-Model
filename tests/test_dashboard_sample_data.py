"""Tests for the bundled sample-data onboarding affordance (issue #1900).

The dashboard previously gated every data-driven page behind a file upload and
never auto-surfaced the bundled ``data/sp500tr_fred_divyield.csv`` series, so a
first-run user saw only empty placeholders. These tests cover the shared helper
that exposes the bundled series and verify it runs end-to-end through the
simulator, which is what the "Use bundled sample data" one-click affordance relies
on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from dashboard.utils import (
    SAMPLE_FINANCING_MODE,
    SAMPLE_INDEX_FILENAME,
    build_sample_model_config,
    bundled_sample_index_path,
    load_bundled_sample_index,
)
from pa_core.config import ModelConfig
from pa_core.data import load_index_returns
from pa_core.orchestrator import SimulatorOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_build_step(argv: list, *, cwd: Path | None = None) -> None:
    """Keep subprocess diagnostics in pytest's failure report, including JUnit."""
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"Packaging command failed (exit {exc.returncode}): {exc.cmd}\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}",
            pytrace=False,
        )


def _build_wheel(destination: Path, source: Path) -> Path:
    environment = destination / "wheel-build-env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    _run_build_step([python, "-m", "pip", "install", "setuptools>=82.0.1", "wheel"])
    wheel_dir = destination / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run_build_step(
        [python, "-m", "pip", "wheel", str(source), "--no-deps", "-w", str(wheel_dir)],
        cwd=destination,
    )
    wheels = sorted(wheel_dir.glob("portable_alpha_extension_model-*.whl"))
    assert wheels, "expected a built project wheel"
    return wheels[-1]


@pytest.mark.parametrize("failing_step", ["install", "wheel"])
def test_build_wheel_retains_failed_command_output(tmp_path, monkeypatch, failing_step):
    """Build failures must retain the cause needed to disposition packaging evidence."""
    monkeypatch.setattr(venv.EnvBuilder, "create", lambda *_args: None)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[3] == failing_step:
            raise subprocess.CalledProcessError(
                1, argv, output="build stage output", stderr="incompatible build environment"
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(pytest.fail.Exception) as caught:
        _build_wheel(tmp_path, tmp_path / "source")
    message = str(caught.value)
    assert "build stage output" in message
    assert "incompatible build environment" in message
    assert str(calls[-1]) in message
    assert "exit 1" in message
    assert len(calls) == (1 if failing_step == "install" else 2)


@pytest.mark.slow
def test_wheel_install_exposes_bundled_dashboard_samples(
    tmp_path: Path, stage_packaging_source: Callable[[Path, Path], Path]
) -> None:
    """Non-editable installs must resolve index, asset, and portfolio samples (#2285)."""
    source = stage_packaging_source(REPO_ROOT, tmp_path / "source")
    wheel_path = _build_wheel(tmp_path, source)
    _assert_installed_wheel_samples(wheel_path, tmp_path)


def test_offline_wheel_install_exposes_bundled_dashboard_samples(
    tmp_path: Path, stage_packaging_source: Callable[[Path, Path], Path]
) -> None:
    """Verify installed product files offline using the interpreter's dependencies.

    This complements the slow fresh-dependency installation above. The isolated
    probe still requires both product packages to come from the new venv.
    """
    source = stage_packaging_source(REPO_ROOT, tmp_path / "source")
    wheel_dir = tmp_path / "wheels"
    _run_build_step(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "wheel",
            str(source),
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "-w",
            str(wheel_dir),
        ],
        cwd=tmp_path,
    )
    (wheel_path,) = wheel_dir.glob("portable_alpha_extension_model-*.whl")
    _assert_installed_wheel_samples(wheel_path, tmp_path, offline=True)


def _assert_installed_wheel_samples(
    wheel_path: Path, tmp_path: Path, *, offline: bool = False
) -> None:
    install_root = tmp_path / "install-env"
    venv.EnvBuilder(with_pip=True, system_site_packages=offline).create(install_root)
    python = install_root / "bin" / "python"
    # An inherited source distribution can make pip report the wheel as already
    # installed even in this fresh venv. Reproduce that metadata-only decoy so
    # installation must ignore PYTHONPATH before the isolated import probe.
    metadata_decoy = tmp_path / "metadata-decoy"
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata_path = metadata_decoy / metadata_name
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_bytes(archive.read(metadata_name))
    install_env = {**os.environ, "PYTHONPATH": str(metadata_decoy)}
    install_options = ["--no-index", "--no-deps", "--ignore-installed"] if offline else []
    subprocess.run(
        [python, "-I", "-m", "pip", "install", *install_options, str(wheel_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=install_env,
    )
    # Poison the inherited module path: a source-tree import must never make
    # this installed-wheel acceptance check pass.
    decoy = tmp_path / "source-decoy"
    package = decoy / "dashboard"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('installed wheel probe imported PYTHONPATH decoy')\n"
    )
    probe_env = {**os.environ, "PYTHONPATH": str(decoy)}
    probe = """
import sys
from pathlib import Path
import dashboard
import pa_core

assert sys.flags.isolated, "wheel probe must ignore inherited source import paths"
for module in (dashboard, pa_core):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), module

from dashboard.utils import (
    bundled_asset_timeseries_path,
    bundled_portfolio_template_path,
    bundled_sample_index_path,
    load_bundled_asset_returns,
    load_bundled_sample_index,
)
from pa_core.schema import load_scenario

index_path = bundled_sample_index_path()
asset_path = bundled_asset_timeseries_path()
portfolio_path = bundled_portfolio_template_path()
for path in (index_path, asset_path, portfolio_path):
    assert path.is_file(), path
    assert path.resolve().is_relative_to(Path(sys.prefix).resolve()), path
assert not load_bundled_sample_index().empty
assert not load_bundled_asset_returns().empty
load_scenario(portfolio_path)
print("ok")
"""
    completed = subprocess.run(
        [python, "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=probe_env,
    )
    assert completed.returncode == 0, (
        f"installed-wheel probe failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "ok" in completed.stdout


def test_bundled_sample_index_path_points_at_repo_data_file() -> None:
    path = bundled_sample_index_path()
    assert path.exists(), f"bundled sample dataset missing at {path}"
    assert path.name == SAMPLE_INDEX_FILENAME
    assert path.parent.name == "data"


def test_load_bundled_sample_index_returns_non_empty_numeric_series() -> None:
    series = load_bundled_sample_index()
    expected = load_index_returns(bundled_sample_index_path())
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert pd.api.types.is_numeric_dtype(series)
    assert not series.isna().any()
    assert series.equals(expected)


def test_bundled_sample_runs_through_orchestrator_end_to_end() -> None:
    """The bundled series must produce a real summary (the one-click demo claim)."""
    index_series = load_bundled_sample_index()
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 100,
            "Number of months": 12,
            "financing_mode": "broadcast",
        }
    )
    orch = SimulatorOrchestrator(cfg, index_series)
    _, summary = orch.run(seed=42)
    assert summary is not None
    assert not summary.empty


def test_sample_run_config_validates() -> None:
    """Stress Lab and Scenario Grid sample config paths must not dead-end."""
    stress_config = build_sample_model_config(
        **{
            "Number of simulations": 1000,
            "Number of months": 12,
            "Total fund capital (mm)": 1000.0,
            "External PA capital (mm)": 200.0,
            "Active Extension capital (mm)": 200.0,
            "Internal PA capital (mm)": 200.0,
            "External PA alpha fraction": 0.5,
            "Active share": 0.5,
        }
    )
    grid_config = build_sample_model_config(
        analysis_mode="alpha_shares",
        external_pa_alpha_min_pct=25.0,
        external_pa_alpha_max_pct=75.0,
        external_pa_alpha_step_pct=5.0,
        active_share_min_pct=20.0,
        active_share_max_pct=100.0,
        active_share_step_pct=5.0,
    )

    assert stress_config.financing_mode == SAMPLE_FINANCING_MODE
    assert grid_config.financing_mode == SAMPLE_FINANCING_MODE
    assert stress_config.financing_mode in {"per_path", "broadcast"}
    assert grid_config.financing_mode in {"per_path", "broadcast"}
    assert grid_config.analysis_mode == "alpha_shares"
    assert grid_config.external_pa_alpha_min_pct == 25.0
    assert grid_config.external_pa_alpha_max_pct == 75.0
    assert grid_config.external_pa_alpha_step_pct == 5.0
    assert grid_config.active_share_min_pct == 20.0
    assert grid_config.active_share_max_pct == 100.0
    assert grid_config.active_share_step_pct == 5.0
