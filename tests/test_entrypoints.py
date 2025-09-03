from __future__ import annotations

import tomllib
from pathlib import Path


def test_pa_validate_entrypoint() -> None:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    scripts = data["project"]["scripts"]
    assert scripts["pa-validate"] == "pa_core.validate:main"


def test_pa_convert_entrypoint() -> None:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    scripts = data["project"]["scripts"]
    assert scripts["pa-convert-params"] == "pa_core.data.convert:main"


def test_pa_entrypoint() -> None:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    scripts = data["project"]["scripts"]
    assert scripts["pa"] == "pa_core.pa:main"
