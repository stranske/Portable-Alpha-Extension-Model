from __future__ import annotations

import warnings
import numpy as npt
from numpy.typing import NDArray

from ..backend import xp as np

__all__ = ["build_cov_matrix"]


def _is_psd(mat: NDArray[npt.float64], tol: float = 0.0) -> bool:
    """Return True if matrix is positive semidefinite within tolerance."""

    eigvals = np.linalg.eigvalsh(mat)
    return float(eigvals.min()) >= -tol


def _nearest_psd(mat: NDArray[npt.float64]) -> NDArray[npt.float64]:
    """Project ``mat`` to the nearest PSD matrix using Higham's method."""

    # Symmetrise input
    sym_mat = 0.5 * (mat + mat.T)
    u, s, vt = np.linalg.svd(sym_mat)
    h = vt.T @ np.diag(s) @ vt

    a2 = 0.5 * (sym_mat + vt.T @ np.diag(s) @ vt)
    a3 = 0.5 * (a2 + a2.T)
    if _is_psd(a3):
        return a3
    # Add jitter until PSD
    spacing = np.spacing(np.linalg.norm(mat))
    eye = np.eye(mat.shape[0])
    k = 1
    while not _is_psd(a3):
        mineig = float(np.min(np.linalg.eigvalsh(a3)))
        a3 += eye * (-mineig * k**2 + spacing)
        k += 1
    return a3


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
) -> NDArray[npt.float64]:
    """Return PSD 4×4 covariance matrix for (Index, H, E, M).

    Volatilities are clipped at zero to avoid negative variances. The
    resulting matrix is symmetrised and, if necessary, projected to the
    nearest positive semidefinite matrix.
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
    cov = 0.5 * (cov + cov.T)
    if _is_psd(cov):
        return cov
    adjusted = _nearest_psd(cov)
    max_delta = float(np.max(np.abs(adjusted - cov)))
    warnings.warn(
        f"Covariance matrix was not PSD; projected with max|Δ|={max_delta:.2e}",
        RuntimeWarning,
    )
    return adjusted
