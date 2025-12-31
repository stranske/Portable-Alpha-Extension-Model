from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..schema import Asset, Correlation


class PortfolioAggregator:
    """Aggregate asset parameters into sleeve-level stats."""

    def __init__(self, assets: list[Asset], correlations: list[Correlation]) -> None:
        self.ids = [a.id for a in assets]
        mu = {a.id: a.mu for a in assets}
        sigma = {a.id: a.sigma for a in assets}
        self.mu_vec = np.array([mu[i] for i in self.ids])
        n = len(self.ids)
        cov = np.eye(n)
        rho_map = {tuple(sorted(c.pair)): c.rho for c in correlations}
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted((self.ids[i], self.ids[j])))
                rho = rho_map.get(pair, 0.0)
                cov[i, j] = cov[j, i] = rho * sigma[self.ids[i]] * sigma[self.ids[j]]
        for i in range(n):
            cov[i, i] = sigma[self.ids[i]] ** 2
        self.cov = cov

    def aggregate(self, weights: Dict[str, float]) -> Tuple[float, float]:
        w = np.array([weights.get(i, 0.0) for i in self.ids])
        mu = float(w @ self.mu_vec)
        sigma = float(np.sqrt(w @ self.cov @ w))
        return mu, sigma

    def cross_corr(self, w_a: Dict[str, float], w_b: Dict[str, float]) -> float:
        w1 = np.array([w_a.get(i, 0.0) for i in self.ids])
        w2 = np.array([w_b.get(i, 0.0) for i in self.ids])
        sigma1 = float(np.sqrt(w1 @ self.cov @ w1))
        sigma2 = float(np.sqrt(w2 @ self.cov @ w2))
        if sigma1 == 0 or sigma2 == 0:
            return 0.0
        cov12 = float(w1 @ self.cov @ w2)
        return cov12 / (sigma1 * sigma2)
