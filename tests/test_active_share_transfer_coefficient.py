"""Tests for the transfer-coefficient (diminishing-returns) active-share fix (#1924).

With decay (kappa) == 0 the agents must reproduce the legacy linear formula
bit-for-bit. With kappa > 0 the expected alpha is concave in the lever (declining
information ratio) while active risk still scales linearly, and an optional
per-share extension cost yields a closed-form interior optimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pa_core.agents.active_ext import ActiveExtensionAgent
from pa_core.agents.external_pa import ExternalPAAgent
from pa_core.agents.types import AgentParams


def _agent(cls, beta_share, extra):
    return cls(AgentParams("X", 100.0, beta_share, 0.0, extra))


def test_kappa_zero_reproduces_legacy_bit_identical():
    rng = np.random.default_rng(0)
    r_beta = rng.normal(0.005, 0.04, size=(50, 12))
    alpha = rng.normal(0.01, 0.02, size=(50, 12))
    fin = np.zeros_like(r_beta)
    beta_share, s = 0.3, 0.6
    legacy = beta_share * (r_beta - fin) + (beta_share * s) * alpha

    # decay=0 with alpha_mu present: tc == 1 -> haircut term is exactly 0.0
    ag = _agent(
        ActiveExtensionAgent,
        beta_share,
        {"active_share": s, "alpha_mu": 0.01, "tc_decay": 0.0, "cost_per_share": 0.0},
    )
    assert np.array_equal(ag.monthly_returns(r_beta, alpha, fin), legacy)

    # legacy extra (no enrichment at all) must also be bit-identical
    ag2 = _agent(ActiveExtensionAgent, beta_share, {"active_share": s})
    assert np.array_equal(ag2.monthly_returns(r_beta, alpha, fin), legacy)


def test_information_ratio_declines_with_active_share():
    rng = np.random.default_rng(1)
    mu, sigma, kappa = 0.01, 0.02, 1.0
    alpha = rng.normal(mu, sigma, size=(20000, 12))
    z = np.zeros_like(alpha)
    irs = []
    for s in (0.2, 0.5, 0.9):
        ag = _agent(
            ActiveExtensionAgent,
            1.0,
            {"active_share": s, "alpha_mu": mu, "tc_decay": kappa, "cost_per_share": 0.0},
        )
        contrib = ag.monthly_returns(z, alpha, z)
        tc = 1.0 / (1.0 + kappa * s)
        # mean scales by s*tc (concave); std scales by s (linear)
        assert contrib.mean() == pytest.approx(s * mu * tc, rel=0.05)
        assert contrib.std() == pytest.approx(s * sigma, rel=0.02)
        irs.append(contrib.mean() / contrib.std())
    assert irs[0] > irs[1] > irs[2]


def test_active_risk_grows_even_as_alpha_saturates():
    rng = np.random.default_rng(2)
    alpha = rng.normal(0.01, 0.02, size=(20000, 12))
    z = np.zeros_like(alpha)
    stds = [
        _agent(
            ActiveExtensionAgent,
            1.0,
            {"active_share": s, "alpha_mu": 0.01, "tc_decay": 1.0, "cost_per_share": 0.0},
        )
        .monthly_returns(z, alpha, z)
        .std()
        for s in (0.2, 0.5, 0.9)
    ]
    assert stds[0] < stds[1] < stds[2]


def test_interior_optimum_matches_closed_form():
    # net expected mean = s*mu/(1+kappa*s) - c*s, peaking at s* = (sqrt(mu/c)-1)/kappa
    mu, sigma, kappa, c = 0.01, 0.0005, 2.0, 0.004
    rng = np.random.default_rng(3)
    alpha = rng.normal(mu, sigma, size=(40000, 6))
    z = np.zeros_like(alpha)
    grid = np.linspace(0.01, 1.0, 100)
    means = [
        _agent(
            ActiveExtensionAgent,
            1.0,
            {"active_share": s, "alpha_mu": mu, "tc_decay": kappa, "cost_per_share": c},
        )
        .monthly_returns(z, alpha, z)
        .mean()
        for s in grid
    ]
    s_opt = grid[int(np.argmax(means))]
    s_star = (np.sqrt(mu / c) - 1.0) / kappa
    assert 0.0 < s_star < 1.0  # sanity: optimum is interior for these params
    assert s_opt == pytest.approx(s_star, abs=0.05)


def test_external_pa_symmetry_kappa_zero_bit_identical():
    rng = np.random.default_rng(4)
    r_beta = rng.normal(0.005, 0.04, size=(50, 12))
    alpha = rng.normal(0.01, 0.02, size=(50, 12))
    fin = np.zeros_like(r_beta)
    beta_share, t = 0.3, 0.6
    legacy = beta_share * (r_beta - fin) + (beta_share * t) * alpha
    ag = _agent(
        ExternalPAAgent,
        beta_share,
        {"theta_extpa": t, "alpha_mu": 0.01, "tc_decay": 0.0, "cost_per_share": 0.0},
    )
    assert np.array_equal(ag.monthly_returns(r_beta, alpha, fin), legacy)


def test_config_wiring_kappa_reduces_active_ext_return():
    """End-to-end: kappa flows ModelConfig -> registry -> agent and lowers ActiveExt alpha."""
    import yaml

    from pa_core.config import load_config
    from pa_core.facade import RunOptions, run_single

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "config/params_template.yml").read_text())
    base["N_SIMULATIONS"] = 4000
    idx = pd.read_csv(root / "data/sp500tr_fred_divyield.csv")["Monthly_TR"].dropna()

    def active_ext_return(overrides):
        d = dict(base)
        d.update(overrides)
        art = run_single(load_config(d), idx, RunOptions(seed=7))
        summ = art.summary.set_index("Agent")
        return float(summ.loc["ActiveExt", "terminal_AnnReturn"])

    r_legacy = active_ext_return({})  # kappa defaults to 0.0
    r_decay = active_ext_return({"active_share_tc_decay": 1.0})
    # same seed => identical draws => the transfer-coefficient haircut deterministically
    # lowers the ActiveExt expected return.
    assert r_decay < r_legacy


def test_preset_example_loads_with_moderate_decay():
    """The shipped 'moderate diminishing returns' example validates and uses kappa=0.43."""
    from pa_core.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "examples/scenarios/active_extension_diminishing_returns.yml"))
    assert cfg.active_share_tc_decay == pytest.approx(0.43)
    assert cfg.theta_tc_decay == pytest.approx(0.43)
    assert cfg.active_ext_cost_per_share > 0.0
