from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

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
        entry.name: entry.value
        for entry in dist.entry_points
        if entry.group == "console_scripts"
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
