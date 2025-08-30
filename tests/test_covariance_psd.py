from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import numpy as np
import pytest

from pa_core.sim.covariance import build_cov_matrix


def test_build_cov_matrix_psd_projection() -> None:
    params = dict(
        rho_idx_H=0.5,
        rho_idx_E=0.5,
        rho_idx_M=0.5,
        rho_H_E=0.5,
        rho_H_M=-0.5,
        rho_E_M=-0.5,
        idx_sigma=0.2,
        sigma_H=0.2,
        sigma_E=0.2,
        sigma_M=0.2,
    )
    sds = np.array([0.2, 0.2, 0.2, 0.2])
    rho = np.array(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.5, -0.5],
            [0.5, 0.5, 1.0, -0.5],
            [0.5, -0.5, -0.5, 1.0],
        ]
    )
    raw_cov = np.outer(sds, sds) * rho
    raw_cov = 0.5 * (raw_cov + raw_cov.T)
    with pytest.warns(RuntimeWarning):
        cov = build_cov_matrix(**params)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-8
    max_delta = float(np.max(np.abs(cov - raw_cov)))
    assert max_delta < 0.03


def test_build_cov_matrix_invalid_rho() -> None:
    with pytest.raises(ValueError):
        build_cov_matrix(
            rho_idx_H=1.2,
            rho_idx_E=0.0,
            rho_idx_M=0.0,
            rho_H_E=0.0,
            rho_H_M=0.0,
            rho_E_M=0.0,
            idx_sigma=0.2,
            sigma_H=0.2,
            sigma_E=0.2,
            sigma_M=0.2,
        )
