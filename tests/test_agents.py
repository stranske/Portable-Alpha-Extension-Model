from pathlib import Path

import numpy as np
import pytest

from pa_core.agents import (
    ActiveExtensionAgent,
    AgentParams,
    BaseAgent,
    ExternalPAAgent,
    InternalBetaAgent,
    InternalPAAgent,
)
from pa_core.agents.registry import build_all, build_from_config, register_agent
from pa_core.config import ModelConfig, normalize_share


def _mock_inputs(shape=(5, 12)):
    r_beta = np.random.normal(size=shape)
    r_H = np.random.normal(size=shape)
    r_E = np.random.normal(size=shape)
    r_M = np.random.normal(size=shape)
    f_int = np.abs(np.random.normal(scale=0.01, size=shape))
    f_ext = np.abs(np.random.normal(scale=0.01, size=shape))
    f_act = np.abs(np.random.normal(scale=0.01, size=shape))
    return r_beta, r_H, r_E, r_M, f_int, f_ext, f_act


class DummyConfigAgent(BaseAgent):
    pass


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
    np.testing.assert_allclose(base.monthly_returns(r_beta, r_H, f), expected_base)

    theta = 0.5
    ext_p = AgentParams("ExternalPA", 100.0, 0.1, 0.0, {"theta_extpa": theta})
    ext = ExternalPAAgent(ext_p)
    expected_ext = ext_p.beta_share * (r_beta - f) + (ext_p.beta_share * theta) * r_M
    np.testing.assert_allclose(ext.monthly_returns(r_beta, r_M, f), expected_ext)

    share_percentage = 70.0  # 70% as percentage value (converted to 0.7 decimal)
    act_p = AgentParams("ActiveExt", 100.0, 0.1, 0.0, {"active_share": share_percentage})
    act = ActiveExtensionAgent(act_p)
    share_fraction = normalize_share(share_percentage)
    expected_act = (
        act_p.beta_share * (r_beta - f) + (act_p.beta_share * float(share_fraction)) * r_E
    )
    np.testing.assert_allclose(act.monthly_returns(r_beta, r_E, f), expected_act)

    beta_p = AgentParams("InternalBeta", 50.0, 1.0, 0.0, {})
    beta_agent = InternalBetaAgent(beta_p)
    expected_beta = beta_p.beta_share * (r_beta - f)
    np.testing.assert_allclose(beta_agent.monthly_returns(r_beta, r_H, f), expected_beta)

    pa_p = AgentParams("InternalPA", 75.0, 0.0, 0.2, {})
    pa_agent = InternalPAAgent(pa_p)
    expected_pa = pa_p.alpha_share * r_H
    np.testing.assert_allclose(pa_agent.monthly_returns(r_beta, r_H, f), expected_pa)


def test_external_pa_theta_zero_returns_pure_beta() -> None:
    """Theta=0 collapses ExternalPAAgent to beta-only sleeve."""

    r_beta, _, _, r_M, _, f_ext, _ = _mock_inputs()
    p = AgentParams("ExternalPA", 100.0, 0.2, 0.0, {"theta_extpa": 0.0})
    agent = ExternalPAAgent(p)
    out = agent.monthly_returns(r_beta, r_M, f_ext)
    expected = p.beta_share * (r_beta - f_ext)
    np.testing.assert_allclose(out, expected)


def test_active_extension_zero_share_returns_pure_beta() -> None:
    """active_share=0 collapses ActiveExtensionAgent to beta-only sleeve."""

    r_beta, _, r_E, _, _, _, f_act = _mock_inputs()
    p = AgentParams("ActiveExt", 100.0, 0.3, 0.0, {"active_share": 0.0})
    agent = ActiveExtensionAgent(p)
    out = agent.monthly_returns(r_beta, r_E, f_act)
    expected = p.beta_share * (r_beta - f_act)
    np.testing.assert_allclose(out, expected)


def test_active_share_fraction_changes_alpha_scale() -> None:
    r_beta = np.zeros((1, 1))
    r_E = np.ones((1, 1))
    f = np.zeros((1, 1))

    high_share = ActiveExtensionAgent(
        AgentParams("ActiveExt", 100.0, 1.0, 0.0, {"active_share": 0.5})
    )
    low_share = ActiveExtensionAgent(
        AgentParams("ActiveExt", 100.0, 1.0, 0.0, {"active_share": 0.005})
    )

    high_out = high_share.monthly_returns(r_beta, r_E, f).item()
    low_out = low_share.monthly_returns(r_beta, r_E, f).item()

    assert high_out == pytest.approx(0.5)
    assert low_out == pytest.approx(0.005)
    assert high_out > low_out * 50.0


def test_base_agent_alpha_zero_returns_pure_beta() -> None:
    r_beta, r_H, *_ = _mock_inputs()
    p = AgentParams("Base", 100.0, 0.7, 0.0, {})
    agent = BaseAgent(p)
    f = np.zeros_like(r_beta)
    out = agent.monthly_returns(r_beta, r_H, f)
    expected = p.beta_share * (r_beta - f)
    np.testing.assert_allclose(out, expected)


def test_internal_pa_no_alpha_returns_zero() -> None:
    """
    When InternalPAAgent is constructed with zero alpha share and receives zero alpha inputs,
    it is expected to return zero monthly returns. This test verifies that the agent
    correctly collapses to a zero-return sleeve in the absence of any alpha contribution.
    """
    r_beta, _, *_ = _mock_inputs()
    zeros = np.zeros_like(r_beta)
    p = AgentParams("InternalPA", 50.0, 0.0, 0.4, {})
    agent = InternalPAAgent(p)
    out = agent.monthly_returns(r_beta, zeros, zeros)
    assert np.allclose(out, 0.0)


def test_build_from_config_basic():
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    agents = build_from_config(cfg)
    names = {type(a).__name__ for a in agents}
    assert "BaseAgent" in names
    assert len(agents) >= 1


def test_build_from_config_generic_agents():
    register_agent("DummyConfig", DummyConfigAgent)
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        agents=[
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.7,
                "alpha_share": 0.3,
                "extra": {},
            },
            {
                "name": "DummyConfig",
                "capital": 100.0,
                "beta_share": 0.2,
                "alpha_share": 0.3,
                "extra": {"note": "ok"},
            },
        ],
    )
    agents = build_from_config(cfg)
    names = {type(a).__name__ for a in agents}
    assert "DummyConfigAgent" in names
    assert "BaseAgent" in names


def test_build_from_config_mixed_agents():
    register_agent("DummyConfigMixed", DummyConfigAgent)
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        external_pa_capital=200.0,
        agents=[
            {
                "name": "DummyConfigMixed",
                "capital": 25.0,
                "beta_share": 0.1,
                "alpha_share": 0.0,
                "extra": {},
            }
        ],
    )
    agents = build_from_config(cfg)
    names = {type(a).__name__ for a in agents}
    assert "BaseAgent" in names
    assert "ExternalPAAgent" in names
    assert "DummyConfigAgent" in names


def test_build_from_config_uses_schedule_margin(tmp_path: Path) -> None:
    schedule_path = tmp_path / "schedule.csv"
    schedule_path.write_text("term,multiplier\n0,2\n1,4\n2,6\n")

    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        financing_model="schedule",
        financing_schedule_path=schedule_path,
        financing_term_months=1.0,
        reference_sigma=0.02,
        total_fund_capital=1000.0,
    )
    agents = build_from_config(cfg)
    internal_beta = next(a for a in agents if isinstance(a, InternalBetaAgent))
    assert internal_beta.p.capital_mm == pytest.approx(80.0)
    assert internal_beta.p.beta_share == pytest.approx(0.08)


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
