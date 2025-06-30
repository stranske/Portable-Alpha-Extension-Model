from hypothesis import given, strategies as st
import numpy as np
from pa_core.simulations import simulate_financing
from pa_core.agents import (AgentParams, BaseAgent, ExternalPAAgent, ActiveExtensionAgent, InternalBetaAgent, InternalPAAgent)

@given(
    T=st.integers(min_value=1, max_value=24),
    n_scenarios=st.integers(min_value=1, max_value=10),
)
def test_simulate_financing_shapes(T, n_scenarios):
    out = simulate_financing(T, 0.0, 0.01, 0.0, 1.0, n_scenarios=n_scenarios)
    expected_shape = (T,) if n_scenarios == 1 else (n_scenarios, T)
    assert out.shape == expected_shape
    assert np.all(np.isfinite(out))
from hypothesis.extra import numpy as nps

@st.composite
def _env(draw):
    n_sim = draw(st.integers(min_value=1, max_value=5))
    n_months = draw(st.integers(min_value=1, max_value=12))
    shape = (n_sim, n_months)
    flt = st.floats(-1.0, 1.0)
    pos = st.floats(0.0, 0.1)
    r_beta = draw(nps.arrays(np.float64, shape, elements=flt))
    r_H = draw(nps.arrays(np.float64, shape, elements=flt))
    r_E = draw(nps.arrays(np.float64, shape, elements=flt))
    r_M = draw(nps.arrays(np.float64, shape, elements=flt))
    f_int = draw(nps.arrays(np.float64, shape, elements=pos))
    f_ext = draw(nps.arrays(np.float64, shape, elements=pos))
    f_act = draw(nps.arrays(np.float64, shape, elements=pos))
    return shape, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act

@st.composite
def _params(draw, name):
    capital = draw(st.floats(min_value=1, max_value=1000))
    beta_share = draw(st.floats(0, 1))
    alpha_share = draw(st.floats(0, 1))
    extra = {}
    if name == "ExternalPA":
        extra["theta_extpa"] = draw(st.floats(0, 1))
    if name == "ActiveExt":
        extra["active_share"] = draw(st.floats(0, 1))
    return AgentParams(name, capital, beta_share, alpha_share, extra)

@given(_env(), _params("Base"))
def test_base_agent_property(env, params):
    shape, r_beta, r_H, *_rest = env
    agent = BaseAgent(params)
    out = agent.monthly_returns(r_beta, r_H, _rest[0])
    assert out.shape == shape
    assert np.all(np.isfinite(out))

@given(_env(), _params("ExternalPA"))
def test_external_pa_agent_property(env, params):
    shape, r_beta, _r_H, _r_E, r_M, _f_int, f_ext, _f_act = env
    agent = ExternalPAAgent(params)
    out = agent.monthly_returns(r_beta, r_M, f_ext)
    assert out.shape == shape
    assert np.all(np.isfinite(out))

@given(_env(), _params("ActiveExt"))
def test_active_ext_agent_property(env, params):
    shape, r_beta, _r_H, r_E, _r_M, _f_int, _f_ext, f_act = env
    agent = ActiveExtensionAgent(params)
    out = agent.monthly_returns(r_beta, r_E, f_act)
    assert out.shape == shape
    assert np.all(np.isfinite(out))

@given(_env(), _params("InternalBeta"))
def test_internal_beta_agent_property(env, params):
    shape, r_beta, r_H, *_rest = env
    agent = InternalBetaAgent(params)
    out = agent.monthly_returns(r_beta, r_H, _rest[2])
    assert out.shape == shape
    assert np.all(np.isfinite(out))

@given(_env(), _params("InternalPA"))
def test_internal_pa_agent_property(env, params):
    shape, _r_beta, r_H, *_rest = env
    agent = InternalPAAgent(params)
    out = agent.monthly_returns(_r_beta, r_H, np.zeros_like(_r_beta))
    assert out.shape == shape
    assert np.all(np.isfinite(out))
