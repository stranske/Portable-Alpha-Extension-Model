import pathlib

import numpy as np

import pa_core.portfolio as portfolio
from pa_core.portfolio import DEFAULT_PORTFOLIO_EXCLUDES, compute_total_contribution_returns


def test_compute_total_contribution_returns_excludes_base_and_total():
    r = np.ones((2, 3))
    returns = {
        "Base": r * 0.1,
        "ExternalPA": r * 0.2,
        "InternalBeta": r * 0.3,
        "Total": r * 9.0,
    }
    total = compute_total_contribution_returns(returns)
    assert total is not None
    assert np.allclose(total, returns["ExternalPA"] + returns["InternalBeta"])


def test_compute_total_contribution_returns_none_when_empty():
    returns = {"Base": np.zeros((1, 1))}
    assert compute_total_contribution_returns(returns) is None


def test_portfolio_is_package_namespace():
    assert hasattr(portfolio, "__path__")
    assert portfolio.__file__ is not None
    assert pathlib.Path(portfolio.__file__).name == "__init__.py"


def test_portfolio_module_is_removed():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert not (repo_root / "pa_core" / "portfolio.py").exists()


def test_portfolio_exports_defaults():
    assert "Base" in DEFAULT_PORTFOLIO_EXCLUDES
