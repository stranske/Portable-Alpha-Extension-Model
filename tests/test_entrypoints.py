from __future__ import annotations

from pathlib import Path

import tomllib


def test_pa_validate_entrypoint() -> None:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    scripts = data["project"]["scripts"]
    assert scripts["pa-validate"] == "pa_core.validate:main"

