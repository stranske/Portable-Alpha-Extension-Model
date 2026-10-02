from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib
import os
import subprocess
import venv
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Callable

import pytest


def _load_scripts() -> dict[str, str]:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    return data["project"]["scripts"]


def _resolve_entrypoint(target: str) -> object:
    module_name, attr_name = target.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def _load_installed_console_scripts() -> dict[str, str]:
    try:
        dist = distribution("portable-alpha-extension-model")
    except PackageNotFoundError:  # pragma: no cover - requires installed package
        pytest.skip("Package metadata not installed.")
    return {
        entry.name: entry.value for entry in dist.entry_points if entry.group == "console_scripts"
    }


def test_pa_validate_entrypoint() -> None:
    scripts = _load_scripts()
    assert scripts["pa-validate"] == "pa_core.validate:main"


def test_pa_convert_entrypoint() -> None:
    scripts = _load_scripts()
    assert scripts["pa-convert-params"] == "pa_core.data.convert:main"


def test_pa_entrypoint() -> None:
    scripts = _load_scripts()
    assert scripts["pa"] == "pa_core.pa:main"


def test_console_scripts_present_and_resolve() -> None:
    scripts = _load_scripts()
    expected = {
        "pa": "pa_core.pa:main",
        "pa-dashboard": "dashboard.cli:main",
        "pa-make-zip": "scripts.make_portable_zip:main",
        "pa-create-launchers": "scripts.create_launchers:main",
    }
    for name, target in expected.items():
        assert scripts.get(name) == target
        resolved = _resolve_entrypoint(target)
        assert callable(resolved)


def test_console_scripts_installed_metadata() -> None:
    scripts = _load_installed_console_scripts()
    expected = {
        "pa": "pa_core.pa:main",
        "pa-dashboard": "dashboard.cli:main",
        "pa-make-zip": "scripts.make_portable_zip:main",
        "pa-create-launchers": "scripts.create_launchers:main",
    }
    for name, target in expected.items():
        assert scripts.get(name) == target


def _venv_python(venv_dir: Path) -> Path:
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    exe = "python.exe" if os.name == "nt" else "python"
    return bin_dir / exe


def _console_script_path(venv_dir: Path, name: str) -> Path:
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    candidates = [
        bin_dir / name,
        bin_dir / f"{name}.exe",
        bin_dir / f"{name}.cmd",
        bin_dir / f"{name}.bat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Console script {name!r} not found in {bin_dir}")


def test_packaging_sources_do_not_share_build_artifacts(
    tmp_path: Path, stage_packaging_source: Callable[[Path, Path], Path]
) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (origin / name).write_text(name)
    for name in ("pa_core", "archive", "scripts", "dashboard", "data", "templates"):
        package = origin / name
        package.mkdir()
        (package / "module.py").write_text(name)
    (origin / "data" / "sample.csv").write_text("value\n1\n")
    (origin / "pa_core" / "build").mkdir()
    (origin / "pa_core" / "build" / "stale.py").write_text("stale")

    first = stage_packaging_source(origin, tmp_path / "first")
    second = stage_packaging_source(origin, tmp_path / "second")
    assert (first / "data" / "sample.csv").read_text() == "value\n1\n"
    assert (second / "pa_core" / "module.py").read_text() == "pa_core"
    assert not (first / "pa_core" / "build").exists()
    assert not (second / "pa_core" / "build").exists()

    (first / "build").mkdir()
    (first / "build" / "stale.py").write_text("first only")
    assert not (second / "build").exists()
    assert not (origin / "build").exists()


def test_console_scripts_work_in_clean_venv(
    tmp_path: Path, stage_packaging_source: Callable[[Path, Path], Path]
) -> None:
    source = stage_packaging_source(Path(__file__).resolve().parents[1], tmp_path / "source")
    venv_dir = tmp_path / "entrypoint-venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)

    # Install setuptools first (not included by default in Python 3.12+ venvs)
    # This is required for --no-build-isolation to work
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "setuptools"],
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        check=True,
    )

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            str(source),
        ],
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        check=True,
        cwd=tmp_path,
    )

    pa = _console_script_path(venv_dir, "pa")
    dashboard = _console_script_path(venv_dir, "pa-dashboard")
    probe_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run([str(pa), "--help"], check=True, cwd=tmp_path, env=probe_env)
    subprocess.run([str(dashboard), "--help"], check=True, cwd=tmp_path, env=probe_env)
