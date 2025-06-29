import numpy as np
from pa_core.simulations import build_cov_matrix, simulate_financing

def test_build_cov_matrix_shape():
    cov = build_cov_matrix(0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)


def test_simulate_financing_shape():
    out = simulate_financing(12, 0.0, 0.01, 0.0, 2.0, n_scenarios=5)
    assert out.shape == (5, 12)
    assert np.all(out >= 0)
