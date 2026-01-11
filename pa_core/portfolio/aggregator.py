from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from ..schema import Asset, Correlation
from ..validators import validate_covariance_matrix_psd


class PortfolioAggregator:
    """Aggregate asset parameters into sleeve-level stats.

    Correlations should be in [-1, 1] and produce a positive semidefinite
    covariance matrix; missing asset pairs default to 0.0 correlation. If the
    covariance is not PSD, it is projected to the nearest PSD matrix with a
    warning. Index correlations should include exactly one non-asset id if
    sleeve_index_corr() is used.
    """

    def __init__(self, assets: list[Asset], correlations: list[Correlation]) -> None:
        self.ids = [a.id for a in assets]
        self._asset_ids = set(self.ids)
        mu = {a.id: a.mu for a in assets}
        sigma = {a.id: a.sigma for a in assets}
        self._sigma = sigma
        self.mu_vec = np.array([mu[i] for i in self.ids])
        n = len(self.ids)
        cov = np.eye(n)
        self._rho_map = {tuple(sorted(c.pair)): c.rho for c in correlations}
        all_corr_ids = {item for pair in self._rho_map for item in pair}
        self._index_ids = sorted(all_corr_ids - self._asset_ids)
        self._index_id = self._index_ids[0] if len(self._index_ids) == 1 else None
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted((self.ids[i], self.ids[j])))
                rho = self._rho_map.get(pair, 0.0)
                cov[i, j] = cov[j, i] = rho * sigma[self.ids[i]] * sigma[self.ids[j]]
        for i in range(n):
            cov[i, i] = sigma[self.ids[i]] ** 2
        validation_result, psd_info = validate_covariance_matrix_psd(cov)
        if psd_info.was_projected:
            warnings.warn(validation_result.message, RuntimeWarning)
            from ..sim.covariance import nearest_psd

            cov = nearest_psd(cov)
        self.cov = cov

    def aggregate(self, weights: Dict[str, float]) -> Tuple[float, float]:
        """Return mean and volatility for a sleeve.

        Weights are keyed by asset id. Correlations should yield a PSD covariance
        matrix to keep the portfolio volatility well-defined.
        """
        w = np.array([weights.get(i, 0.0) for i in self.ids])
        mu = float(w @ self.mu_vec)
        sigma = float(np.sqrt(w @ self.cov @ w))
        return mu, sigma

    def cross_corr(self, w_a: Dict[str, float], w_b: Dict[str, float]) -> float:
        """Return correlation between two sleeves."""
        w1 = np.array([w_a.get(i, 0.0) for i in self.ids])
        w2 = np.array([w_b.get(i, 0.0) for i in self.ids])
        sigma1 = float(np.sqrt(w1 @ self.cov @ w1))
        sigma2 = float(np.sqrt(w2 @ self.cov @ w2))
        if sigma1 == 0 or sigma2 == 0:
            return 0.0
        cov12 = float(w1 @ self.cov @ w2)
        return cov12 / (sigma1 * sigma2)

    def sleeve_index_corr(self, weights: Dict[str, float], index_sigma: float) -> float:
        """Return correlation between a sleeve and the benchmark index.

        Correlations must include index-to-asset pairs for all sleeve assets.
        index_sigma is the benchmark volatility; if it or the sleeve volatility
        is zero, the correlation is defined as 0.0.
        """
        if self._index_id is None:
            if not self._index_ids:
                raise ValueError("index id not found in correlations")
            raise ValueError(f"ambiguous index ids in correlations: {self._index_ids}")
        missing = [
            asset_id
            for asset_id in self.ids
            if tuple(sorted((self._index_id, asset_id))) not in self._rho_map
        ]
        if missing:
            raise ValueError(f"missing index correlations for assets: {sorted(missing)}")
        w = np.array([weights.get(i, 0.0) for i in self.ids])
        sleeve_sigma = float(np.sqrt(w @ self.cov @ w))
        if sleeve_sigma == 0.0 or index_sigma == 0.0:
            return 0.0
        cov_sleeve_index = 0.0
        for asset_id in self.ids:
            rho = self._rho_map[tuple(sorted((self._index_id, asset_id)))]
            cov_sleeve_index += (
                weights.get(asset_id, 0.0) * rho * index_sigma * self._sigma[asset_id]
            )
        return cov_sleeve_index / (sleeve_sigma * index_sigma)
