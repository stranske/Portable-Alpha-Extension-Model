from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

import pytest

import types
import sys

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.schema import Scenario, load_scenario, save_scenario


def test_roundtrip(tmp_path: Path) -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"A": 1.0}}],
    }
    scen = Scenario.model_validate(data)
    out = tmp_path / "scen.yaml"
    save_scenario(scen, out)
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


def test_duplicate_correlations() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [
            {"pair": ["IDX", "A"], "rho": 0.1},
            {"pair": ["A", "IDX"], "rho": 0.1},
        ],
        "portfolios": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        Scenario.model_validate(data)


def test_sleeve_capital_share_sum() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [],
        "sleeves": {
            "s1": {"alpha_source": "p:1", "capital_share": 0.6},
            "s2": {"alpha_source": "p:2", "capital_share": 0.5},
        },
    }
    with pytest.raises(ValueError, match="capital_share"):
        Scenario.model_validate(data)


def test_duplicate_asset_ids() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [
            {"id": "A", "mu": 0.05, "sigma": 0.1},
            {"id": "A", "mu": 0.04, "sigma": 0.1},
        ],
        "correlations": [],
        "portfolios": [],
    }
    with pytest.raises(ValueError, match="duplicate asset ids"):
        Scenario.model_validate(data)


def test_duplicate_portfolio_ids() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [
            {"id": "p1", "weights": {"A": 1.0}},
            {"id": "p1", "weights": {"A": 1.0}},
        ],
    }
    with pytest.raises(ValueError, match="duplicate portfolio ids"):
        Scenario.model_validate(data)


def test_portfolio_unknown_asset() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [{"pair": ["IDX", "A"], "rho": 0.1}],
        "portfolios": [{"id": "p1", "weights": {"B": 1.0}}],
    }
    with pytest.raises(ValueError, match="unknown assets"):
        Scenario.model_validate(data)


def test_extra_correlations() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [
            {"pair": ["IDX", "A"], "rho": 0.1},
            {"pair": ["IDX", "B"], "rho": 0.2},
        ],
        "portfolios": [],
    }
    with pytest.raises(ValueError, match="unexpected correlations"):
        Scenario.model_validate(data)
