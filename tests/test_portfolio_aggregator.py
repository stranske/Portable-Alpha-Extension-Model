from __future__ import annotations

# ruff: noqa: E402

import numpy as np
import pytest
import types
import sys
from pathlib import Path

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.portfolio import PortfolioAggregator
from pa_core.schema import Asset, Correlation


def test_aggregator_single_and_cross() -> None:
    assets = [
        Asset(id="A", label="A", mu=0.1, sigma=0.2),
        Asset(id="B", label="B", mu=0.05, sigma=0.1),
    ]
    corrs = [Correlation(pair=("A", "B"), rho=0.25)]
    agg = PortfolioAggregator(assets, corrs)

    mu, sigma = agg.aggregate({"A": 1.0})
    assert mu == pytest.approx(0.1)
    assert sigma == pytest.approx(0.2)

    w = {"A": 0.6, "B": 0.4}
    mu2, sigma2 = agg.aggregate(w)
    mu_vec = np.array([0.1, 0.05])
    cov = np.array([[0.04, 0.25 * 0.2 * 0.1], [0.25 * 0.2 * 0.1, 0.01]])
    expected_mu = float(np.array([0.6, 0.4]) @ mu_vec)
    expected_sigma = float(np.sqrt(np.array([0.6, 0.4]) @ cov @ np.array([0.6, 0.4])))
    assert mu2 == pytest.approx(expected_mu)
    assert sigma2 == pytest.approx(expected_sigma)

    corr_ab = agg.cross_corr({"A": 1.0}, {"B": 1.0})
    assert corr_ab == pytest.approx(0.25)
