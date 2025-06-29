import numpy as np
from pa_core.agents import (
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    ActiveExtensionAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from pa_core.agents.registry import build_all


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
