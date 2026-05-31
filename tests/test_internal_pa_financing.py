"""Tests for internal-PA financing cost support (issue #1849).

Covers the synthetic index financing curves, the financing-series resolver,
the InternalPA agent net-of-financing return, routing through
``simulate_agents``, and an end-to-end ``run_single`` sensitivity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_core.agents.internal_pa import InternalPAAgent
from pa_core.agents.types import AgentParams
from pa_core.config import AgentConfig, load_config
from pa_core.data.index_financing_curves import (
    INDEX_FINANCING_CURVES_BPS,
    annual_bps_to_monthly,
    available_indices,
    get_index_financing_curve_monthly,
)
from pa_core.facade import RunOptions, run_single
from pa_core.random import spawn_agent_rngs, spawn_rngs
from pa_core.reporting.attribution import compute_sleeve_return_attribution
from pa_core.sim import resolve_internal_pa_financing_series
from pa_core.simulations import simulate_agents
from pa_core.sweep import run_parameter_sweep

# ---------------------------------------------------------------------------
# Index financing curve fixture
# ---------------------------------------------------------------------------


def test_annual_bps_to_monthly() -> None:
    # 120 bps annual -> 0.012 annual -> 0.001 monthly
    assert annual_bps_to_monthly(120.0) == pytest.approx(0.001)
    # negative spread converts symmetrically (positive carry / benefit)
    assert annual_bps_to_monthly(-120.0) == pytest.approx(-0.001)


def test_index_curve_supports_negative_spreads() -> None:
    assert any(
        v < 0 for v in INDEX_FINANCING_CURVES_BPS.values()
    ), "fixture must include at least one negative (positive-carry) curve"
    curve = get_index_financing_curve_monthly("NKY", 4)
    assert len(curve) == 4
    assert all(c == curve[0] for c in curve)
    assert curve[0] < 0  # NKY is an illustrative negative-cost curve


def test_index_curve_available_and_unknown() -> None:
    assert "SPX" in available_indices()
    # case-insensitive lookup
    assert get_index_financing_curve_monthly("spx", 2) == get_index_financing_curve_monthly(
        "SPX", 2
    )
    with pytest.raises(KeyError):
        get_index_financing_curve_monthly("NOT_AN_INDEX", 3)
    with pytest.raises(ValueError):
        get_index_financing_curve_monthly("SPX", 0)


# ---------------------------------------------------------------------------
# Financing-series resolver
# ---------------------------------------------------------------------------


def test_resolve_defaults_to_zero() -> None:
    mat = resolve_internal_pa_financing_series(n_months=3, n_sim=5)
    assert mat.shape == (5, 3)
    assert np.allclose(mat, 0.0)


def test_resolve_deterministic_mean_allows_negative() -> None:
    pos = resolve_internal_pa_financing_series(n_months=3, n_sim=2, mean_month=0.01)
    assert np.allclose(pos, 0.01)
    neg = resolve_internal_pa_financing_series(n_months=3, n_sim=2, mean_month=-0.01)
    assert np.allclose(neg, -0.01)  # not clipped at zero


def test_resolve_explicit_series() -> None:
    mat = resolve_internal_pa_financing_series(n_months=3, n_sim=4, series=[0.01, -0.02, 0.0])
    assert mat.shape == (4, 3)
    assert np.allclose(mat[0], [0.01, -0.02, 0.0])
    # broadcast identically across simulations
    assert np.allclose(mat, mat[0])


def test_resolve_series_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        resolve_internal_pa_financing_series(n_months=3, n_sim=2, series=[0.01, 0.02])


def test_resolve_index_curve() -> None:
    mat = resolve_internal_pa_financing_series(n_months=2, n_sim=3, index="SPX")
    expected = annual_bps_to_monthly(INDEX_FINANCING_CURVES_BPS["SPX"])
    assert mat.shape == (3, 2)
    assert np.allclose(mat, expected)


def test_resolve_stochastic_not_clipped() -> None:
    rng = np.random.default_rng(0)
    mat = resolve_internal_pa_financing_series(
        n_months=240,
        n_sim=1,
        mean_month=-0.02,
        sigma_month=0.01,
        financing_mode="per_path",
        rng=rng,
    )
    # A strongly negative mean must keep negative values (no zero-clipping).
    assert float(mat.min()) < 0.0


# ---------------------------------------------------------------------------
# InternalPA agent
# ---------------------------------------------------------------------------


def _ones(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape)


def test_internal_pa_agent_subtracts_financing() -> None:
    shape = (2, 3)
    agent = InternalPAAgent(AgentParams("InternalPA", 100.0, 0.0, 1.0, {}))
    r_beta = _ones(shape) * 0.01
    alpha = _ones(shape) * 0.02
    no_fin = agent.monthly_returns(r_beta, alpha, np.zeros(shape))
    pos_fin = agent.monthly_returns(r_beta, alpha, np.full(shape, 0.005))
    neg_fin = agent.monthly_returns(r_beta, alpha, np.full(shape, -0.005))
    assert np.allclose(no_fin, 0.02)
    assert np.allclose(pos_fin, 0.015)  # positive cost lowers return
    assert np.allclose(neg_fin, 0.025)  # negative cost (carry) raises return


# ---------------------------------------------------------------------------
# simulate_agents routing
# ---------------------------------------------------------------------------


def test_simulate_agents_routes_financing_to_internal_pa_only() -> None:
    n_sim, n_months = 3, 4
    rng = np.random.default_rng(1)
    r_beta = rng.normal(size=(n_sim, n_months))
    r_H = rng.normal(size=(n_sim, n_months))
    r_E = rng.normal(size=(n_sim, n_months))
    r_M = rng.normal(size=(n_sim, n_months))
    f = np.zeros((n_sim, n_months))
    f_int_pa = np.full((n_sim, n_months), 0.01)
    agents = [
        InternalPAAgent(AgentParams("InternalPA", 100.0, 0.0, 1.0, {})),
    ]
    with_fin = simulate_agents(agents, r_beta, r_H, r_E, r_M, f, f, f, f_int_pa)
    # default (None) -> zeros -> backward-compatible pure alpha
    without_fin = simulate_agents(agents, r_beta, r_H, r_E, r_M, f, f, f)
    assert np.allclose(without_fin["InternalPA"], r_H)  # alpha_share=1.0
    assert np.allclose(with_fin["InternalPA"], r_H - 0.01)


# ---------------------------------------------------------------------------
# End-to-end sensitivity via run_single (acceptance criteria)
# ---------------------------------------------------------------------------


def _internal_pa_scenario():
    return load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 200,
            "N_MONTHS": 6,
            "agents": [
                AgentConfig(
                    name="InternalPA",
                    capital=150.0,
                    beta_share=0.0,
                    alpha_share=1.0,
                )
            ],
        }
    )


def _internal_pa_mean(cfg) -> float:
    idx = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01])
    artifacts = run_single(cfg, idx, RunOptions(seed=7))
    return float(artifacts.raw_returns["InternalPA"].to_numpy().mean())


def test_run_single_internal_pa_financing_sensitivity() -> None:
    base = _internal_pa_scenario()
    zero_fin = _internal_pa_mean(base.model_copy(update={"internal_pa_financing_mean_month": 0.0}))
    pos_fin = _internal_pa_mean(base.model_copy(update={"internal_pa_financing_mean_month": 0.01}))
    neg_fin = _internal_pa_mean(base.model_copy(update={"internal_pa_financing_mean_month": -0.01}))

    # AC: InternalPA return changes when internal financing cost changes,
    # holding alpha inputs fixed; positive lowers, negative raises.
    assert pos_fin < zero_fin
    assert neg_fin > zero_fin
    # Deterministic financing (sigma=0) shifts the mean by exactly the cost.
    assert zero_fin - pos_fin == pytest.approx(0.01, abs=1e-9)
    assert neg_fin - zero_fin == pytest.approx(0.01, abs=1e-9)


def test_run_single_accepts_index_curve() -> None:
    cfg = _internal_pa_scenario().model_copy(update={"internal_pa_financing_index": "SPX"})
    mean = _internal_pa_mean(cfg)
    assert np.isfinite(mean)


def test_parameter_sweep_passes_internal_pa_financing_series(monkeypatch) -> None:
    cfg = _internal_pa_scenario().model_copy(
        update={
            "N_SIMULATIONS": 200,
            "N_MONTHS": 6,
            "analysis_mode": "returns",
            "internal_pa_financing_mean_month": 0.01,
            "in_house_return_min_pct": 1.0,
            "in_house_return_max_pct": 1.0,
            "in_house_return_step_pct": 1.0,
            "in_house_vol_min_pct": 1.0,
            "in_house_vol_max_pct": 1.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_min_pct": 1.0,
            "alpha_ext_return_max_pct": 1.0,
            "alpha_ext_return_step_pct": 1.0,
            "alpha_ext_vol_min_pct": 1.0,
            "alpha_ext_vol_max_pct": 1.0,
            "alpha_ext_vol_step_pct": 1.0,
        }
    )
    idx = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01])
    captured: dict[str, np.ndarray] = {}

    def fake_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros

    def fake_simulate_agents(*args):
        captured["internal_pa_financing"] = args[-1]
        return {"Base": np.zeros((cfg.N_SIMULATIONS, cfg.N_MONTHS))}

    monkeypatch.setattr("pa_core.sweep.draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr("pa_core.sweep.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sweep.simulate_agents", fake_simulate_agents)

    run_parameter_sweep(
        cfg,
        idx,
        spawn_rngs(7, 1)[0],
        spawn_agent_rngs(7, ["internal", "external_pa", "active_ext"]),
    )

    assert captured["internal_pa_financing"].shape == (cfg.N_SIMULATIONS, cfg.N_MONTHS)
    assert np.allclose(captured["internal_pa_financing"], 0.01)


def test_attribution_includes_internal_pa_financing_row() -> None:
    cfg = _internal_pa_scenario().model_copy(update={"internal_pa_financing_mean_month": 0.01})
    idx = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01])
    df = compute_sleeve_return_attribution(cfg, idx)
    int_rows = df[df["Agent"] == "InternalPA"]
    assert "Financing" in set(int_rows["Sub"])
    fin_row = int_rows[int_rows["Sub"] == "Financing"]["Return"].iloc[0]
    assert fin_row < 0  # positive cost shows as a negative contribution
