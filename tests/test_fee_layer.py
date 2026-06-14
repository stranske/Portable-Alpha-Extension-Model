"""Tests for the per-sleeve management/performance fee layer (issue #1904).

Covers the ``FeeSchedule`` model and validation, the ``compute_fee_drag`` /
``apply_fees`` helpers, routing through ``simulate_agents``, an end-to-end
``run_single`` sensitivity check, and config plumbing (default + round-trip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pa_core.agents.internal_pa import InternalPAAgent
from pa_core.agents.types import AgentParams
from pa_core.config import AgentConfig, load_config
from pa_core.facade import RunOptions, run_single
from pa_core.fees import FeeSchedule, apply_fees, compute_fee_drag
from pa_core.simulations import simulate_agents

# ---------------------------------------------------------------------------
# FeeSchedule model
# ---------------------------------------------------------------------------


def test_default_schedule_is_zero() -> None:
    sched = FeeSchedule()
    assert sched.mgmt_fee_bps == 0.0
    assert sched.perf_fee_pct == 0.0
    assert sched.hurdle_bps == 0.0
    assert sched.is_zero


def test_schedule_with_fee_is_not_zero() -> None:
    assert not FeeSchedule(mgmt_fee_bps=50.0).is_zero
    assert not FeeSchedule(perf_fee_pct=0.2).is_zero


def test_negative_or_out_of_range_fee_rejected() -> None:
    with pytest.raises(ValidationError):
        FeeSchedule(mgmt_fee_bps=-1.0)
    with pytest.raises(ValidationError):
        FeeSchedule(perf_fee_pct=-0.1)
    with pytest.raises(ValidationError):
        FeeSchedule(perf_fee_pct=1.5)
    with pytest.raises(ValidationError):
        FeeSchedule(hurdle_bps=-5.0)


def test_schedule_coerces_from_dict() -> None:
    sched = FeeSchedule.model_validate(
        {"mgmt_fee_bps": 50.0, "perf_fee_pct": 0.2, "hurdle_bps": 200.0}
    )
    assert sched.mgmt_fee_bps == 50.0
    assert sched.perf_fee_pct == 0.2
    assert sched.hurdle_bps == 200.0


# ---------------------------------------------------------------------------
# compute_fee_drag / apply_fees
# ---------------------------------------------------------------------------


def test_mgmt_fee_constant_monthly_drag() -> None:
    gross = np.full((2, 3), 0.01)
    # 120 bps annual -> 0.012 annual -> 0.001 monthly flat drag
    sched = FeeSchedule(mgmt_fee_bps=120.0)
    drag = compute_fee_drag(gross, sched)
    assert np.allclose(drag, 0.001)
    assert np.allclose(apply_fees(gross, sched), 0.009)


def test_perf_fee_charged_above_hurdle_only() -> None:
    gross = np.array([[0.02, -0.01, 0.0]])
    sched = FeeSchedule(perf_fee_pct=0.5, hurdle_bps=0.0)  # 50% of positive return
    drag = compute_fee_drag(gross, sched)
    # No rebate / fee on returns at or below the (zero) hurdle.
    assert np.allclose(drag, [[0.01, 0.0, 0.0]])


def test_perf_fee_respects_hurdle() -> None:
    gross = np.full((1, 1), 0.01)
    # hurdle 120 bps annual -> 0.001 monthly; 100% perf on the excess above it.
    sched = FeeSchedule(perf_fee_pct=1.0, hurdle_bps=120.0)
    drag = compute_fee_drag(gross, sched)
    assert np.allclose(drag, 0.01 - 0.001)


def test_zero_schedule_is_noop() -> None:
    gross = np.full((2, 2), 0.03)
    # is_zero short-circuits and returns the same object untouched.
    assert apply_fees(gross, FeeSchedule()) is gross


# ---------------------------------------------------------------------------
# simulate_agents routing
# ---------------------------------------------------------------------------


def test_simulate_agents_applies_fee_to_named_sleeve_only() -> None:
    n_sim, n_months = 3, 4
    rng = np.random.default_rng(2)
    r_beta = rng.normal(size=(n_sim, n_months))
    r_H = np.full((n_sim, n_months), 0.02)
    f = np.zeros((n_sim, n_months))
    agents = [InternalPAAgent(AgentParams("InternalPA", 100.0, 0.0, 1.0, {}))]

    gross = simulate_agents(agents, r_beta, r_H, r_H, r_H, f, f, f)
    net = simulate_agents(
        agents,
        r_beta,
        r_H,
        r_H,
        r_H,
        f,
        f,
        f,
        fee_schedule={"InternalPA": FeeSchedule(mgmt_fee_bps=120.0)},
    )
    assert np.allclose(gross["InternalPA"], 0.02)
    assert np.allclose(net["InternalPA"], 0.019)  # exactly the 0.001 mgmt drag

    # A schedule for a sleeve that is not present is a no-op.
    untouched = simulate_agents(
        agents,
        r_beta,
        r_H,
        r_H,
        r_H,
        f,
        f,
        f,
        fee_schedule={"ExternalPA": FeeSchedule(mgmt_fee_bps=999.0)},
    )
    assert np.allclose(untouched["InternalPA"], 0.02)


def test_simulate_agents_default_is_backward_compatible() -> None:
    n_sim, n_months = 2, 3
    r_beta = np.zeros((n_sim, n_months))
    r_H = np.full((n_sim, n_months), 0.02)
    f = np.zeros((n_sim, n_months))
    agents = [InternalPAAgent(AgentParams("InternalPA", 100.0, 0.0, 1.0, {}))]
    # No fee_schedule arg -> identical to gross.
    assert np.allclose(simulate_agents(agents, r_beta, r_H, r_H, r_H, f, f, f)["InternalPA"], 0.02)


# ---------------------------------------------------------------------------
# End-to-end via run_single (acceptance criteria) + config plumbing
# ---------------------------------------------------------------------------


def _scenario():
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


def test_default_config_has_no_fee_schedule() -> None:
    assert _scenario().fee_schedule is None


def test_run_single_mgmt_fee_lowers_return_by_exact_drag() -> None:
    base = _scenario()
    gross = _internal_pa_mean(base)
    net = _internal_pa_mean(
        base.model_copy(update={"fee_schedule": {"InternalPA": FeeSchedule(mgmt_fee_bps=120.0)}})
    )
    # Deterministic mgmt fee shifts the mean by exactly the monthly drag (0.001).
    assert gross - net == pytest.approx(0.001, abs=1e-9)


def test_run_single_perf_fee_lowers_return() -> None:
    base = _scenario()
    gross = _internal_pa_mean(base)
    net = _internal_pa_mean(
        base.model_copy(
            update={"fee_schedule": {"InternalPA": FeeSchedule(perf_fee_pct=0.5, hurdle_bps=0.0)}}
        )
    )
    assert net < gross


def test_fee_schedule_survives_model_dump() -> None:
    cfg = _scenario().model_copy(
        update={"fee_schedule": {"InternalPA": FeeSchedule(mgmt_fee_bps=50.0)}}
    )
    dumped = cfg.model_dump()
    assert dumped["fee_schedule"]["InternalPA"]["mgmt_fee_bps"] == 50.0
