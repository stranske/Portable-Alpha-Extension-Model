import numpy as np

from pa_core.agents import (
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
)
from pa_core.random import spawn_agent_rngs
from pa_core.sim.covariance import build_cov_matrix
from pa_core.sim.paths import draw_financing_series
from pa_core.simulations import simulate_agents, simulate_financing
from pa_core.validators import SYNTHETIC_DATA_MEAN, SYNTHETIC_DATA_STD


def test_build_cov_matrix_shape():
    cov = build_cov_matrix(0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)


def test_simulate_financing_shape():
    out = simulate_financing(
        12, SYNTHETIC_DATA_MEAN, SYNTHETIC_DATA_STD, 0.0, 2.0, n_scenarios=5
    )
    assert out.shape == (5, 12)
    assert np.all(out >= 0)


def test_build_cov_matrix_near_singular():
    cov = build_cov_matrix(0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.2, 0.2, 0.2, 0.2)
    # matrix should remain symmetric
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)


def test_simulate_financing_spikes():
    out = simulate_financing(6, SYNTHETIC_DATA_MEAN, 1.0, 1.0, 5.0, seed=42)
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


def test_draw_financing_series_rngs():
    params = {
        "internal_financing_mean_month": 0.0,
        "internal_financing_sigma_month": 0.01,
        "internal_spike_prob": 0.0,
        "internal_spike_factor": 0.0,
        "ext_pa_financing_mean_month": 0.0,
        "ext_pa_financing_sigma_month": 0.01,
        "ext_pa_spike_prob": 0.0,
        "ext_pa_spike_factor": 0.0,
        "act_ext_financing_mean_month": 0.0,
        "act_ext_financing_sigma_month": 0.01,
        "act_ext_spike_prob": 0.0,
        "act_ext_spike_factor": 0.0,
    }
    rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    out1 = draw_financing_series(n_months=3, n_sim=2, params=params, rngs=rngs)
    rngs2 = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    out2 = draw_financing_series(n_months=3, n_sim=2, params=params, rngs=rngs2)
    for a, b in zip(out1, out2):
        assert np.allclose(a, b)
