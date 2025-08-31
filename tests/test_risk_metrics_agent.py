from __future__ import annotations

# ruff: noqa: E402

import numpy as np
import pytest

from pa_core.agents.risk_metrics import RiskMetricsAgent
from pa_core.sim.metrics import (
    breach_count,
    breach_probability,
    conditional_value_at_risk,
    max_drawdown,
    time_under_water,
)


# Test data fixtures for reproducible random data generation

@pytest.fixture
def standard_returns_data() -> np.ndarray:
    """Standard returns data for basic agent testing."""
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility
    return rng.normal(0.0, 0.1, size=(100, 12))


@pytest.fixture
def small_returns_data() -> np.ndarray:
    """Smaller returns dataset for scaling tests."""
    rng = np.random.default_rng(1)  # Different seed for variety
    return rng.normal(0.0, 0.05, size=(50, 6))


@pytest.fixture
def agent_with_custom_params() -> RiskMetricsAgent:
    """Risk metrics agent with custom parameters for specific testing."""
    return RiskMetricsAgent(var_conf=0.95, breach_threshold=-0.02)


@pytest.fixture  
def agent_default_params() -> RiskMetricsAgent:
    """Risk metrics agent with default parameters."""
    return RiskMetricsAgent()


# Test for normal functionality - agent matches individual functions

def test_risk_metrics_agent_matches_individual_functions(standard_returns_data, agent_with_custom_params) -> None:
    """Test that RiskMetricsAgent produces same results as individual metric functions."""
    returns = standard_returns_data
    agent = agent_with_custom_params
    metrics = agent.run(returns)
    
    # Verify each metric matches corresponding function
    assert metrics.cvar == pytest.approx(
        conditional_value_at_risk(returns, confidence=0.95)
    )
    assert metrics.max_drawdown == pytest.approx(max_drawdown(returns))
    assert metrics.time_under_water == pytest.approx(time_under_water(returns))
    assert metrics.breach_probability == pytest.approx(
        breach_probability(returns, -0.02)
    )
    assert metrics.breach_count == breach_count(returns, -0.02)


# Test for scaling behavior - parametrized for multiple scaling factors

@pytest.mark.parametrize("scale_factor", [
    1.5,  # Moderate scaling
    2.0,  # Double scaling
    3.0,  # Triple scaling
])
def test_risk_metrics_agent_scaling_behavior(small_returns_data, agent_default_params, scale_factor) -> None:
    """Test that risk metrics scale appropriately when returns are scaled."""
    base_returns = small_returns_data
    scaled_returns = scale_factor * base_returns
    agent = agent_default_params
    
    base_metrics = agent.run(base_returns)
    scaled_metrics = agent.run(scaled_returns)
    
    # For negative values (losses), scaling should increase the magnitude
    # abs(scaled) should be >= scale_factor * abs(base) for risk metrics
    assert abs(scaled_metrics.cvar) >= abs(scale_factor * base_metrics.cvar) * 0.9  # Allow 10% tolerance for numerical effects
    assert abs(scaled_metrics.max_drawdown) >= abs(scale_factor * base_metrics.max_drawdown) * 0.9


# Edge cases for problematic scenarios  

@pytest.mark.parametrize("returns_shape,expected_behavior", [
    ((10, 12), "normal"),     # Small dataset
    ((1000, 12), "normal"),   # Large dataset  
    ((100, 1), "normal"),     # Single period
])
def test_risk_metrics_agent_different_data_sizes(returns_shape, expected_behavior, agent_default_params) -> None:
    """Test that agent handles different data sizes appropriately."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.1, size=returns_shape)
    agent = agent_default_params
    
    # Should run without error for different sizes
    metrics = agent.run(returns)
    
    # Basic sanity checks
    assert isinstance(metrics.cvar, float)
    assert isinstance(metrics.max_drawdown, float) 
    assert isinstance(metrics.time_under_water, float)
    assert isinstance(metrics.breach_probability, float)
    assert isinstance(metrics.breach_count, int)
