from __future__ import annotations

from .backend import xp as np
from numpy.typing import NDArray

__all__ = ["build_cov_matrix"]


def build_cov_matrix(
    rho_idx_H: float,
    rho_idx_E: float,
    rho_idx_M: float,
    rho_H_E: float,
    rho_H_M: float,
    rho_E_M: float,
    idx_sigma: float,
    sigma_H: float,
    sigma_E: float,
    sigma_M: float,
) -> NDArray[np.float64]:
    """Return 4×4 covariance matrix for (Index, H, E, M).

    Volatilities are clipped at zero to avoid negative variances and the
    resulting matrix is symmetrised to guard against numerical drift.
    """
    sds = np.clip(np.array([idx_sigma, sigma_H, sigma_E, sigma_M]), 0.0, None)
    rho = np.array(
        [
            [1.0, rho_idx_H, rho_idx_E, rho_idx_M],
            [rho_idx_H, 1.0, rho_H_E, rho_H_M],
            [rho_idx_E, rho_H_E, 1.0, rho_E_M],
            [rho_idx_M, rho_H_M, rho_E_M, 1.0],
        ]
    )
    cov = np.outer(sds, sds) * rho
    return 0.5 * (cov + cov.T)
