from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pa_core.schema import (
    ASSET_INDEX_CONFLICT_ERROR,
    Scenario,
    load_scenario,
    save_scenario,
)

# ruff: noqa: E402


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


def test_index_only_allows_empty_correlations() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [],
    }
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


def test_duplicate_correlations_same_order() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [
            {"pair": ["IDX", "A"], "rho": 0.1},
            {"pair": ["IDX", "A"], "rho": 0.1},
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


def test_sleeve_share_bounds() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [],
        "sleeves": {
            "s1": {"alpha_source": "p:1", "capital_share": 120, "active_share": 0.5},
        },
    }
    with pytest.raises(ValueError, match="capital_share must be between 0 and 1"):
        Scenario.model_validate(data)


def test_sleeve_share_normalization() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [],
        "correlations": [],
        "portfolios": [],
        "sleeves": {
            "s1": {"alpha_source": "p:1", "capital_share": 60, "theta": 50},
            "s2": {"alpha_source": "p:2", "capital_share": 40, "active_share": 20},
        },
    }
    scenario = Scenario.model_validate(data)
    sleeves = scenario.sleeves or {}
    assert sleeves["s1"].capital_share == pytest.approx(0.6)
    assert sleeves["s1"].theta == pytest.approx(0.5)
    assert sleeves["s2"].capital_share == pytest.approx(0.4)
    assert sleeves["s2"].active_share == pytest.approx(0.2)


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


def test_self_pair_correlation_is_unexpected() -> None:
    data = {
        "index": {"id": "IDX", "mu": 0.1, "sigma": 0.2},
        "assets": [{"id": "A", "mu": 0.05, "sigma": 0.1}],
        "correlations": [
            {"pair": ["IDX", "A"], "rho": 0.1},
            {"pair": ["A", "A"], "rho": 0.0},
        ],
        "portfolios": [],
    }
    with pytest.raises(ValueError, match="unexpected correlations"):
        Scenario.model_validate(data)


def test_scenario_rejects_index_in_assets() -> None:
    expected = ASSET_INDEX_CONFLICT_ERROR
    with pytest.raises(ValidationError) as excinfo:
        Scenario.model_validate(
            {
                "index": {"id": "IDX", "mu": 0.05, "sigma": 0.1},
                "assets": [{"id": "IDX", "mu": 0.04, "sigma": 0.08}],
                "correlations": [{"pair": ["IDX", "IDX"], "rho": 0.0}],
            }
        )
    errors = excinfo.value.errors()
    assert errors[0]["msg"] == f"Value error, {expected}"
    assert errors[0]["type"] == "value_error"
