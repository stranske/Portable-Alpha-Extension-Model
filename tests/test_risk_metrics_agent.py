from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import types
import sys

import numpy as np
import pytest

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.agents.risk_metrics import RiskMetricsAgent
from pa_core.sim.metrics import (
    breach_count,
    breach_probability,
    conditional_value_at_risk,
    max_drawdown,
    time_under_water,
)


def test_risk_metrics_agent_matches_functions() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.1, size=(100, 12))
    agent = RiskMetricsAgent(var_conf=0.95, breach_threshold=-0.02)
    metrics = agent.run(returns)
    assert metrics.cvar == pytest.approx(
        conditional_value_at_risk(returns, confidence=0.95)
    )
    assert metrics.max_drawdown == pytest.approx(max_drawdown(returns))
    assert metrics.time_under_water == pytest.approx(time_under_water(returns))
    assert metrics.breach_probability == pytest.approx(
        breach_probability(returns, -0.02)
    )
    assert metrics.breach_count == breach_count(returns, -0.02)


def test_risk_metrics_agent_scaling() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.05, size=(50, 6))
    scaled = 2 * base
    agent = RiskMetricsAgent()
    m1 = agent.run(base)
    m2 = agent.run(scaled)
    assert m2.cvar <= 2 * m1.cvar
    assert m2.max_drawdown <= 2 * m1.max_drawdown
