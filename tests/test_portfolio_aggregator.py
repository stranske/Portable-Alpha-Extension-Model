from __future__ import annotations

import numpy as np
import pytest

from pa_core.portfolio import PortfolioAggregator
from pa_core.schema import Asset, Correlation

# ruff: noqa: E402


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


def test_aggregator_projects_non_psd_covariance() -> None:
    assets = [
        Asset(id="A", label="A", mu=0.1, sigma=1.0),
        Asset(id="B", label="B", mu=0.05, sigma=1.0),
        Asset(id="C", label="C", mu=0.02, sigma=1.0),
    ]
    corrs = [
        Correlation(pair=("A", "B"), rho=0.9),
        Correlation(pair=("A", "C"), rho=0.9),
        Correlation(pair=("B", "C"), rho=-0.9),
    ]
    with pytest.warns(RuntimeWarning, match="Projected to PSD"):
        agg = PortfolioAggregator(assets, corrs)
    eigvals = np.linalg.eigvalsh(agg.cov)
    assert eigvals.min() >= -1e-10


def test_sleeve_index_corr() -> None:
    assets = [
        Asset(id="A", label="A", mu=0.1, sigma=0.2),
        Asset(id="B", label="B", mu=0.05, sigma=0.1),
    ]
    corrs = [
        Correlation(pair=("A", "B"), rho=0.25),
        Correlation(pair=("IDX", "A"), rho=0.5),
        Correlation(pair=("IDX", "B"), rho=-0.2),
    ]
    agg = PortfolioAggregator(assets, corrs)
    weights = {"A": 0.6, "B": 0.4}
    index_sigma = 0.15
    w = np.array([0.6, 0.4])
    cov = np.array([[0.04, 0.25 * 0.2 * 0.1], [0.25 * 0.2 * 0.1, 0.01]])
    sleeve_sigma = float(np.sqrt(w @ cov @ w))
    cov_sleeve_index = 0.6 * 0.5 * 0.2 * index_sigma + 0.4 * -0.2 * 0.1 * index_sigma
    expected = cov_sleeve_index / (sleeve_sigma * index_sigma)
    assert agg.sleeve_index_corr(weights, index_sigma) == pytest.approx(expected)


def test_sleeve_index_corr_requires_index_pairs() -> None:
    assets = [
        Asset(id="A", label="A", mu=0.1, sigma=0.2),
        Asset(id="B", label="B", mu=0.05, sigma=0.1),
    ]
    corrs = [Correlation(pair=("A", "B"), rho=0.25)]
    agg = PortfolioAggregator(assets, corrs)
    with pytest.raises(ValueError, match="index id not found"):
        agg.sleeve_index_corr({"A": 1.0}, 0.1)
