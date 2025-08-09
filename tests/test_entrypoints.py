from __future__ import annotations

from pathlib import Path

import tomllib


def test_pa_validate_entrypoint() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["pa-validate"] == "pa_core.validate:main"

