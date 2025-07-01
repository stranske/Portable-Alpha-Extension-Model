import numpy as np
import pytest
from pa_core.agents import (
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    ActiveExtensionAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from pa_core.agents.registry import build_all, build_from_config
from pa_core.config import ModelConfig


def _mock_inputs(shape=(5, 12)):
    r_beta = np.random.normal(size=shape)
    r_H = np.random.normal(size=shape)
    r_E = np.random.normal(size=shape)
    r_M = np.random.normal(size=shape)
    f_int = np.abs(np.random.normal(scale=0.01, size=shape))
    f_ext = np.abs(np.random.normal(scale=0.01, size=shape))
    f_act = np.abs(np.random.normal(scale=0.01, size=shape))
    return r_beta, r_H, r_E, r_M, f_int, f_ext, f_act


def test_base_agent_shape():
    r_beta, r_H, *_ = _mock_inputs()
    p = AgentParams("Base", 100.0, 0.5, 0.5, {})
    agent = BaseAgent(p)
    out = agent.monthly_returns(r_beta, r_H, np.zeros_like(r_beta))
    assert out.shape == r_beta.shape
    assert np.all(np.isfinite(out))


def test_external_pa_agent_shape():
    r_beta, _, _, r_M, _, f_ext, _ = _mock_inputs()
    p = AgentParams("ExternalPA", 200.0, 0.1, 0.0, {"theta_extpa": 0.3})
    agent = ExternalPAAgent(p)
    out = agent.monthly_returns(r_beta, r_M, f_ext)
    assert out.shape == r_beta.shape


def test_active_extension_agent_shape():
    r_beta, _, r_E, _, _, _, f_act = _mock_inputs()
    p = AgentParams("ActiveExt", 150.0, 0.1, 0.0, {"active_share": 0.5})
    agent = ActiveExtensionAgent(p)
    out = agent.monthly_returns(r_beta, r_E, f_act)
    assert out.shape == r_beta.shape


def test_internal_agents_shape():
    r_beta, r_H, *_ = _mock_inputs()
    int_beta = InternalBetaAgent(AgentParams("InternalBeta", 50.0, 0.05, 0.0, {}))
    int_pa = InternalPAAgent(AgentParams("InternalPA", 75.0, 0.0, 0.05, {}))
    out_beta = int_beta.monthly_returns(r_beta, r_H, np.zeros_like(r_beta))
    out_pa = int_pa.monthly_returns(r_beta, r_H, np.zeros_like(r_beta))
    assert out_beta.shape == r_beta.shape
    assert out_pa.shape == r_beta.shape


def test_registry_build_all():
    params_list = [
        AgentParams("Base", 100.0, 0.5, 0.5, {}),
        AgentParams("ExternalPA", 100.0, 0.1, 0.0, {}),
    ]
    agents = build_all(params_list)
    assert len(agents) == 2
    assert isinstance(agents[0], BaseAgent)
    assert isinstance(agents[1], ExternalPAAgent)


def test_agent_math_identity():
    r_beta = np.array([[0.01, -0.02]])
    r_H = np.array([[0.03, 0.04]])
    r_E = np.array([[0.05, -0.01]])
    r_M = np.array([[0.02, 0.01]])
    f = np.array([[0.001, 0.002]])

    base_p = AgentParams("Base", 100.0, 0.6, 0.4, {})
    base = BaseAgent(base_p)
    expected_base = base_p.beta_share * (r_beta - f) + base_p.alpha_share * r_H
    np.testing.assert_allclose(
        base.monthly_returns(r_beta, r_H, f), expected_base
    )

    theta = 0.5
    ext_p = AgentParams("ExternalPA", 100.0, 0.1, 0.0, {"theta_extpa": theta})
    ext = ExternalPAAgent(ext_p)
    expected_ext = ext_p.beta_share * (r_beta - f) + (
        ext_p.beta_share * theta
    ) * r_M
    np.testing.assert_allclose(ext.monthly_returns(r_beta, r_M, f), expected_ext)

    share = 0.7
    act_p = AgentParams("ActiveExt", 100.0, 0.1, 0.0, {"active_share": share})
    act = ActiveExtensionAgent(act_p)
    expected_act = act_p.beta_share * (r_beta - f) + (
        act_p.beta_share * share
    ) * r_E
    np.testing.assert_allclose(act.monthly_returns(r_beta, r_E, f), expected_act)

    beta_p = AgentParams("InternalBeta", 50.0, 1.0, 0.0, {})
    beta_agent = InternalBetaAgent(beta_p)
    expected_beta = beta_p.beta_share * (r_beta - f)
    np.testing.assert_allclose(
        beta_agent.monthly_returns(r_beta, r_H, f), expected_beta
    )

    pa_p = AgentParams("InternalPA", 75.0, 0.0, 0.2, {})
    pa_agent = InternalPAAgent(pa_p)
    expected_pa = pa_p.alpha_share * r_H
    np.testing.assert_allclose(pa_agent.monthly_returns(r_beta, r_H, f), expected_pa)


def test_build_from_config_basic():
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)
    agents = build_from_config(cfg)
    names = {type(a).__name__ for a in agents}
    assert "BaseAgent" in names
    assert len(agents) >= 1


def test_dimension_mismatch_errors():
    r_beta = np.zeros((2, 2))
    r_H = np.zeros((2, 3))  # mismatched shape
    f = np.zeros((2, 2))

    base = BaseAgent(AgentParams("Base", 1.0, 0.5, 0.5, {}))
    try:
        base.monthly_returns(r_beta, r_H, f)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for shape mismatch")

    beta = InternalBetaAgent(AgentParams("InternalBeta", 1.0, 1.0, 0.0, {}))
    try:
        beta.monthly_returns(r_beta, r_H, f)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for shape mismatch")

    pa = InternalPAAgent(AgentParams("InternalPA", 1.0, 0.0, 0.1, {}))
    try:
        pa.monthly_returns(r_beta, r_H, f)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for shape mismatch")


def test_dimension_invalid_ndim():
    base = BaseAgent(AgentParams("Base", 1.0, 0.5, 0.5, {}))

    bad3d = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        base.monthly_returns(bad3d, bad3d, bad3d)

    bad1d = np.zeros(5)
    with pytest.raises(ValueError):
        base.monthly_returns(bad1d, bad1d, bad1d)
