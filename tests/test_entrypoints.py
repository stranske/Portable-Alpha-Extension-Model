from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import subprocess
import venv

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


def test_console_scripts_work_in_clean_venv(tmp_path: Path) -> None:
    venv_dir = tmp_path / "entrypoint-venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            ".",
        ],
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        check=True,
    )

    pa = _console_script_path(venv_dir, "pa")
    dashboard = _console_script_path(venv_dir, "pa-dashboard")
    subprocess.run([str(pa), "--help"], check=True)
    subprocess.run([str(dashboard), "--help"], check=True)
