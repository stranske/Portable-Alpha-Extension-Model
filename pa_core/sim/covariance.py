from __future__ import annotations

import warnings
import numpy as npt
from numpy.typing import NDArray

from ..backend import xp as np
from ..schema import CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND

__all__ = ["build_cov_matrix", "nearest_psd"]


def _is_psd(mat: NDArray[npt.float64], tol: float = 0.0) -> bool:
    """Return True if matrix is positive semidefinite within tolerance."""

    eigvals = np.linalg.eigvalsh(mat)
    return eigvals.min() >= -tol


def nearest_psd(mat: NDArray[npt.float64]) -> NDArray[npt.float64]:
    """Return the nearest positive semidefinite matrix using Higham's method.

    The input is symmetrised and eigenvalues are clipped at zero. If the
    resulting matrix is still not PSD, jitter is added iteratively until all
    eigenvalues are non-negative.
    """

    # Symmetrise input
    sym_mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym_mat)
    eigvals_clipped = np.clip(eigvals, 0, None)
    psd_mat = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    a3 = 0.5 * (psd_mat + psd_mat.T)
    if _is_psd(a3):
        return a3
    # Add jitter until PSD
    spacing = np.spacing(np.linalg.norm(mat))
    eye = np.eye(mat.shape[0])
    k = 1
    while not _is_psd(a3):
        eigvals = np.linalg.eigvalsh(a3)
        mineig = float(eigvals.min())
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

    for name, rho in [
        ("rho_idx_H", rho_idx_H),
        ("rho_idx_E", rho_idx_E),
        ("rho_idx_M", rho_idx_M),
        ("rho_H_E", rho_H_E),
        ("rho_H_M", rho_H_M),
        ("rho_E_M", rho_E_M),
    ]:
        if not (CORRELATION_LOWER_BOUND <= rho <= CORRELATION_UPPER_BOUND):
            raise ValueError(
                f"{name} must be between {CORRELATION_LOWER_BOUND} and {CORRELATION_UPPER_BOUND}"
            )

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
    adjusted = nearest_psd(cov)
    max_delta = float(np.max(np.abs(adjusted - cov)))
    warnings.warn(
        f"Covariance matrix was not PSD; projected with max|Δ|={max_delta:.2e}",
        RuntimeWarning,
    )
    return adjusted
