import numpy as np
from pa_core.covariance import build_cov_matrix
from pa_core.simulations import simulate_financing, simulate_agents
from pa_core.agents import (
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
)

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


def test_simulate_agents_vectorised():
    n_sim, n_months = 4, 3
    r_beta = np.random.normal(size=(n_sim, n_months))
    r_H = np.random.normal(size=(n_sim, n_months))
    r_E = np.random.normal(size=(n_sim, n_months))
    r_M = np.random.normal(size=(n_sim, n_months))
    f = np.abs(np.random.normal(size=(n_sim, n_months))) * 0.01
    agents = [
        BaseAgent(AgentParams("Base", 100, 0.5, 0.5, {})),
        ExternalPAAgent(AgentParams("ExternalPA", 80, 0.2, 0.0, {})),
        InternalBetaAgent(AgentParams("InternalBeta", 20, 1.0, 0.0, {})),
    ]
    results = simulate_agents(agents, r_beta, r_H, r_E, r_M, f, f, f)
    assert set(results) == {"Base", "ExternalPA", "InternalBeta"}
    for arr in results.values():
        assert arr.shape == (n_sim, n_months)
