from __future__ import annotations

# ruff: noqa: E402

import types
import sys
from pathlib import Path

import numpy as np

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)
import numpy as np
from pa_core.sim.covariance import nearest_psd


def test_nearest_psd_makes_matrix_psd() -> None:
    mat = np.array([[1.0, 2.0], [2.0, 1.0]])
    psd = nearest_psd(mat)
    eigvals = np.linalg.eigvalsh(psd)
    assert eigvals.min() >= -1e-8
    assert np.allclose(psd, psd.T)


def test_nearest_psd_leaves_psd_matrix_unchanged() -> None:
    mat = np.eye(3)
    psd = nearest_psd(mat)
    assert np.allclose(psd, mat)
