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


def test_build_cov_matrix_near_singular():
    cov = build_cov_matrix(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.2, 0.2, 0.2, 0.2)
    # matrix should remain symmetric
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)


def test_simulate_financing_spikes():
    out = simulate_financing(6, 0.0, 1.0, 1.0, 5.0, seed=42)
    # with spike_prob=1 all months should include spike component
    assert np.all(out >= 0)
    assert np.all(out > 0)
