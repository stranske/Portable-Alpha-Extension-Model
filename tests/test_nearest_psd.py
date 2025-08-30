from __future__ import annotations

# ruff: noqa: E402

import numpy as np
from pa_core.sim.covariance import nearest_psd

# Tolerance for numerical precision errors when checking if eigenvalues are non-negative
# in PSD matrix validation. Eigenvalues slightly below zero (e.g., > -1e-8) are considered
# acceptable due to floating-point inaccuracies.
EIGENVALUE_TOLERANCE = -1e-8


def test_nearest_psd_makes_matrix_psd() -> None:
    mat = np.array([[1.0, 2.0], [2.0, 1.0]])
    psd = nearest_psd(mat)
    eigvals = np.linalg.eigvalsh(psd)
    assert eigvals.min() >= EIGENVALUE_TOLERANCE
    assert np.allclose(psd, psd.T)


def test_nearest_psd_leaves_psd_matrix_unchanged() -> None:
    mat = np.eye(3)
    psd = nearest_psd(mat)
    assert np.allclose(psd, mat)
