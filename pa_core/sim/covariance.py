from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..backend import xp
from ..schema import CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND

__all__ = ["build_cov_matrix", "nearest_psd", "build_cov_matrix_with_validation"]


def _boost_shrinkage_for_short_samples(
    shrinkage: float, *, n_samples: int, n_features: int
) -> float:
    """Boost shrinkage when sample sizes are short to limit PSD adjustments."""

    min_samples = max(12, 2 * n_features)
    if n_samples >= min_samples:
        return shrinkage
    ratio = n_samples / float(min_samples)
    boosted = 1.0 - (1.0 - shrinkage) * ratio
    return float(min(1.0, max(shrinkage, boosted)))


def _ledoit_wolf_shrinkage_from_cov(
    cov: NDArray[np.float64], n_samples: int
) -> tuple[NDArray[np.float64], float]:
    """Return Ledoit-Wolf shrunk covariance and shrinkage intensity.

    Uses a normal-moment approximation to estimate shrinkage intensity from
    the covariance matrix and sample count.
    """

    if n_samples <= 1:
        return np.asarray(cov, dtype=np.float64), 0.0
    n_features = cov.shape[0]
    mu = float(xp.trace(cov)) / n_features
    target = mu * xp.eye(n_features)
    delta = cov - target
    delta_norm2 = float(xp.sum(delta**2))
    if delta_norm2 == 0.0:
        return np.asarray(cov, dtype=np.float64), 0.0
    diag = xp.diag(cov)
    beta = float(xp.sum(cov**2) + xp.sum(xp.outer(diag, diag))) / float(n_samples)
    beta = min(beta, delta_norm2)
    shrinkage = 0.0 if delta_norm2 == 0.0 else beta / delta_norm2
    shrinkage = _boost_shrinkage_for_short_samples(
        shrinkage, n_samples=n_samples, n_features=n_features
    )
    shrunk_cov = (1.0 - shrinkage) * cov + shrinkage * target
    return np.asarray(shrunk_cov, dtype=np.float64), float(shrinkage)


def _is_psd(mat: NDArray[np.float64], tol: float = 0.0) -> bool:
    """Return True if matrix is positive semidefinite within tolerance."""

    eigvals = xp.linalg.eigvalsh(mat)
    return bool(eigvals.min() >= -tol)


def nearest_psd(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the nearest positive semidefinite matrix using Higham's method.

    The input is symmetrised and eigenvalues are clipped at zero. If the
    resulting matrix is still not PSD, jitter is added iteratively until all
    eigenvalues are non-negative.
    """

    # Symmetrise input
    sym_mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = xp.linalg.eigh(sym_mat)
    eigvals_clipped = xp.clip(eigvals, 0, None)
    psd_mat = eigvecs @ xp.diag(eigvals_clipped) @ eigvecs.T
    a3 = 0.5 * (psd_mat + psd_mat.T)
    if _is_psd(a3):
        return np.asarray(a3, dtype=np.float64)
    # Add jitter until PSD
    spacing = xp.spacing(xp.linalg.norm(mat))
    eye = xp.eye(mat.shape[0])
    k = 1
    while not _is_psd(a3):
        eigvals = xp.linalg.eigvalsh(a3)
        mineig = float(eigvals.min())
        a3 += eye * (-mineig * k**2 + spacing)
        k += 1
    return np.asarray(a3, dtype=np.float64)


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
    covariance_shrinkage: Literal["none", "ledoit_wolf"] = "none",
    n_samples: int | None = None,
) -> NDArray[np.float64]:
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

    sds = xp.clip(xp.array([idx_sigma, sigma_H, sigma_E, sigma_M]), 0.0, None)
    rho = xp.array(
        [
            [1.0, rho_idx_H, rho_idx_E, rho_idx_M],
            [rho_idx_H, 1.0, rho_H_E, rho_H_M],
            [rho_idx_E, rho_H_E, 1.0, rho_E_M],
            [rho_idx_M, rho_H_M, rho_E_M, 1.0],
        ]
    )
    cov = xp.outer(sds, sds) * rho
    cov = 0.5 * (cov + cov.T)
    if covariance_shrinkage == "ledoit_wolf" and n_samples is not None:
        cov, _ = _ledoit_wolf_shrinkage_from_cov(cov, int(n_samples))
        cov = xp.asarray(cov)
    if _is_psd(cov):
        return np.asarray(cov, dtype=np.float64)
    adjusted = nearest_psd(cov)
    max_delta = float(xp.max(xp.abs(adjusted - cov)))
    warnings.warn(
        f"Covariance matrix was not PSD; projected with max|Δ|={max_delta:.2e}",
        RuntimeWarning,
    )
    return np.asarray(adjusted, dtype=np.float64)


def build_cov_matrix_with_validation(
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
    covariance_shrinkage: Literal["none", "ledoit_wolf"] = "none",
    n_samples: int | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Return PSD 4×4 covariance matrix for (Index, H, E, M) with detailed validation info.

    Similar to build_cov_matrix but returns additional validation information
    for enhanced user feedback.

    Returns:
        Tuple of (covariance_matrix, validation_info)
        where validation_info contains details about any PSD projection.
    """
    from ..validators import validate_correlations, validate_covariance_matrix_psd

    correlations = {
        "rho_idx_H": rho_idx_H,
        "rho_idx_E": rho_idx_E,
        "rho_idx_M": rho_idx_M,
        "rho_H_E": rho_H_E,
        "rho_H_M": rho_H_M,
        "rho_E_M": rho_E_M,
    }

    # Validate correlations
    correlation_results = validate_correlations(correlations)

    # Check for any correlation validation errors
    has_errors = any(not r.is_valid for r in correlation_results)
    if has_errors:
        error_msgs = [r.message for r in correlation_results if not r.is_valid]
        raise ValueError("; ".join(error_msgs))

    # Build matrix using existing logic
    sds = xp.clip(xp.array([idx_sigma, sigma_H, sigma_E, sigma_M]), 0.0, None)
    rho = xp.array(
        [
            [1.0, rho_idx_H, rho_idx_E, rho_idx_M],
            [rho_idx_H, 1.0, rho_H_E, rho_H_M],
            [rho_idx_E, rho_H_E, 1.0, rho_E_M],
            [rho_idx_M, rho_H_M, rho_E_M, 1.0],
        ]
    )
    cov = xp.outer(sds, sds) * rho
    cov = 0.5 * (cov + cov.T)
    if covariance_shrinkage == "ledoit_wolf" and n_samples is not None:
        cov, _ = _ledoit_wolf_shrinkage_from_cov(cov, int(n_samples))
        cov = xp.asarray(cov)

    # Validate PSD and get detailed info
    validation_result, psd_info = validate_covariance_matrix_psd(cov)

    validation_info = {
        "correlation_validations": correlation_results,
        "psd_validation": validation_result,
        "psd_info": psd_info,
        "was_projected": psd_info.was_projected,
    }

    if psd_info.was_projected:
        # Issue warning and return projected matrix
        warnings.warn(validation_result.message, RuntimeWarning)
        return nearest_psd(cov), validation_info
    else:
        return np.asarray(cov, dtype=np.float64), validation_info
