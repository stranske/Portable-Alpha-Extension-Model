import numpy as np
import pytest

from pa_core.agents import AgentParams, BaseAgent, ExternalPAAgent, InternalBetaAgent
from pa_core.config import ModelConfig
from pa_core.random import spawn_agent_rngs
from pa_core.sim.covariance import build_cov_matrix
from pa_core.sim.params import build_simulation_params
from pa_core.sim.paths import draw_financing_series
from pa_core.simulations import simulate_agents, simulate_financing
from pa_core.validators import SYNTHETIC_DATA_MEAN, SYNTHETIC_DATA_STD


def _build_financing_params(**updates: float) -> dict[str, float | str | None]:
    base_cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        return_unit="monthly",
        mu_H=0.0,
        sigma_H=0.0,
        mu_E=0.0,
        sigma_E=0.0,
        mu_M=0.0,
        sigma_M=0.0,
        internal_financing_mean_month=0.0,
        internal_financing_sigma_month=0.0,
        internal_spike_prob=0.0,
        internal_spike_factor=0.0,
        ext_pa_financing_mean_month=0.0,
        ext_pa_financing_sigma_month=0.0,
        ext_pa_spike_prob=0.0,
        ext_pa_spike_factor=0.0,
        act_ext_financing_mean_month=0.0,
        act_ext_financing_sigma_month=0.0,
        act_ext_spike_prob=0.0,
        act_ext_spike_factor=0.0,
    )
    cfg = base_cfg.model_copy(update=updates)
    return build_simulation_params(cfg, mu_idx=0.0, idx_sigma=0.0)


def test_build_cov_matrix_shape():
    cov = build_cov_matrix(0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)


def test_simulate_financing_shape():
    out = simulate_financing(12, SYNTHETIC_DATA_MEAN, SYNTHETIC_DATA_STD, 0.0, 2.0, n_scenarios=5)
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


def test_simulate_financing_varies_with_sigma():
    out = simulate_financing(12, SYNTHETIC_DATA_MEAN, 0.5, 0.0, 2.0, seed=7)
    assert np.std(out) > 0


def test_simulate_financing_spike_shift():
    n_months = 8
    mean = 10.0
    sigma = 0.2
    spike_factor = 3.0
    base = simulate_financing(n_months, mean, sigma, 0.0, spike_factor, seed=11)
    spiked = simulate_financing(n_months, mean, sigma, 1.0, spike_factor, seed=11)
    assert np.allclose(spiked - base, spike_factor * sigma)
    assert np.std(spiked) > 0


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
    assert set(results) == {"Base", "ExternalPA", "InternalBeta", "Total"}
    for arr in results.values():
        assert arr.shape == (n_sim, n_months)


def test_total_returns_sum_weighted_sleeves():
    n_sim, n_months = 2, 2
    r_beta = np.full((n_sim, n_months), 0.01)
    r_H = np.full((n_sim, n_months), 0.02)
    r_E = np.full((n_sim, n_months), 0.03)
    r_M = np.full((n_sim, n_months), 0.04)
    f = np.zeros((n_sim, n_months))
    agents = [
        BaseAgent(AgentParams("Base", 100, 0.5, 0.5, {})),
        ExternalPAAgent(AgentParams("ExternalPA", 200, 0.3, 0.0, {"theta_extpa": 0.5})),
        InternalBetaAgent(AgentParams("InternalBeta", 300, 0.4, 0.0, {})),
    ]
    results = simulate_agents(agents, r_beta, r_H, r_E, r_M, f, f, f)
    total = results["Total"]
    expected = results["ExternalPA"] + results["InternalBeta"]
    assert np.allclose(total, expected)


def test_draw_financing_series_broadcasts_monthly_vector():
    params = _build_financing_params(
        internal_financing_mean_month=0.02,
        internal_financing_sigma_month=0.05,
        internal_spike_prob=0.0,
        internal_spike_factor=0.0,
        ext_pa_financing_mean_month=0.02,
        ext_pa_financing_sigma_month=0.05,
        ext_pa_spike_prob=0.0,
        ext_pa_spike_factor=0.0,
        act_ext_financing_mean_month=0.02,
        act_ext_financing_sigma_month=0.05,
        act_ext_spike_prob=0.0,
        act_ext_spike_factor=0.0,
    )
    rng = np.random.default_rng(17)
    f_int, f_ext, f_act = draw_financing_series(
        n_months=6,
        n_sim=3,
        params=params,
        financing_mode="broadcast",
        rng=rng,
    )
    for mat in (f_int, f_ext, f_act):
        assert mat.shape == (3, 6)
        assert np.allclose(mat[0], mat[1])
        assert np.allclose(mat[1], mat[2])
    assert np.std(f_int[0]) > 0


def test_draw_financing_series_rngs():
    params = _build_financing_params(
        internal_financing_mean_month=0.0,
        internal_financing_sigma_month=0.01,
        internal_spike_prob=0.0,
        internal_spike_factor=0.0,
        ext_pa_financing_mean_month=0.0,
        ext_pa_financing_sigma_month=0.01,
        ext_pa_spike_prob=0.0,
        ext_pa_spike_factor=0.0,
        act_ext_financing_mean_month=0.0,
        act_ext_financing_sigma_month=0.01,
        act_ext_spike_prob=0.0,
        act_ext_spike_factor=0.0,
    )
    rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    out1 = draw_financing_series(
        n_months=3,
        n_sim=2,
        params=params,
        financing_mode="broadcast",
        rngs=rngs,
    )
    rngs2 = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    out2 = draw_financing_series(
        n_months=3,
        n_sim=2,
        params=params,
        financing_mode="broadcast",
        rngs=rngs2,
    )
    for a, b in zip(out1, out2):
        assert np.allclose(a, b)


def test_draw_financing_series_per_path_varies_by_simulation():
    params = {
        "internal_financing_mean_month": 0.01,
        "internal_financing_sigma_month": 0.05,
        "internal_spike_prob": 0.0,
        "internal_spike_factor": 0.0,
        "ext_pa_financing_mean_month": 0.01,
        "ext_pa_financing_sigma_month": 0.05,
        "ext_pa_spike_prob": 0.0,
        "ext_pa_spike_factor": 0.0,
        "act_ext_financing_mean_month": 0.01,
        "act_ext_financing_sigma_month": 0.05,
        "act_ext_spike_prob": 0.0,
        "act_ext_spike_factor": 0.0,
    }
    rng = np.random.default_rng(42)
    f_int, f_ext, f_act = draw_financing_series(
        n_months=48,
        n_sim=3,
        params=params,
        financing_mode="per_path",
        rng=rng,
    )
    for mat in (f_int, f_ext, f_act):
        assert mat.shape == (3, 48)
        assert not np.allclose(mat[0], mat[1])
        corr = np.corrcoef(mat[0], mat[1])[0, 1]
        assert corr < 0.99


def test_financing_mode_correlation_structure():
    params = _build_financing_params(
        internal_financing_mean_month=0.01,
        internal_financing_sigma_month=0.05,
        internal_spike_prob=0.0,
        internal_spike_factor=0.0,
        ext_pa_financing_mean_month=0.01,
        ext_pa_financing_sigma_month=0.05,
        ext_pa_spike_prob=0.0,
        ext_pa_spike_factor=0.0,
        act_ext_financing_mean_month=0.01,
        act_ext_financing_sigma_month=0.05,
        act_ext_spike_prob=0.0,
        act_ext_spike_factor=0.0,
    )

    rng = np.random.default_rng(123)
    f_int_broadcast, _, _ = draw_financing_series(
        n_months=120,
        n_sim=2,
        params=params,
        financing_mode="broadcast",
        rng=rng,
    )
    assert np.std(f_int_broadcast[0]) > 0
    corr_broadcast = np.corrcoef(f_int_broadcast[0], f_int_broadcast[1])[0, 1]
    assert corr_broadcast == pytest.approx(1.0)

    rng = np.random.default_rng(123)
    f_int_per_path, _, _ = draw_financing_series(
        n_months=240,
        n_sim=2,
        params=params,
        financing_mode="per_path",
        rng=rng,
    )
    corr_per_path = np.corrcoef(f_int_per_path[0], f_int_per_path[1])[0, 1]
    assert abs(corr_per_path) < 0.3
