from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

import pytest
import yaml
import types
import sys

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.validate import main


def test_validate_cli(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    path = tmp_path / "scen.yaml"
    path.write_text(yaml.safe_dump(data))
    main([str(path)])


def test_validate_cli_fail(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [{"id": "p1", "weights": {"IDX": 0.5}}],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        main([str(path)])
