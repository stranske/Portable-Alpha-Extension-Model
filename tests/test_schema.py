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

from pa_core.schema import Scenario, load_scenario


def test_roundtrip(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    scen = Scenario.model_validate(data)
    out = tmp_path / "scen.yaml"
    out.write_text(yaml.safe_dump(scen.model_dump()))
    scen2 = load_scenario(out)
    assert scen2 == scen


def test_bad_portfolio_weights() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [{"id": "p1", "weights": {"IDX": 0.6}}],
    }
    with pytest.raises(ValueError):
        Scenario.model_validate(data)


def test_missing_rho() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    with pytest.raises(ValueError):
        Scenario.model_validate(data)
