# ruff: noqa: E402
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.validators import TEST_TOLERANCE_EPSILON


# Test data fixtures for clear separation of normal vs problematic cases

@pytest.fixture
def valid_config_basic() -> Dict[str, Any]:
    """Basic valid configuration for standard orchestrator tests."""
    return {
        "N_SIMULATIONS": 10,
        "N_MONTHS": 5,
        "w_beta_H": 0.6,
        "w_alpha_H": 0.4,
        "risk_metrics": ["ShortfallProb"],
    }


@pytest.fixture  
def valid_config_no_alpha() -> Dict[str, Any]:
    """Valid configuration with no alpha allocation for TE=0 testing."""
    return {
        "N_SIMULATIONS": 5,
        "N_MONTHS": 12,
        "w_beta_H": 1.0,
        "w_alpha_H": 0.0,
        "external_pa_capital": 0.0,
        "active_ext_capital": 0.0,
        "internal_pa_capital": 0.0,
        "risk_metrics": ["ShortfallProb"],
    }


@pytest.fixture
def valid_config_reproducibility() -> Dict[str, Any]:
    """Valid configuration for reproducibility testing."""
    return {
        "N_SIMULATIONS": 5,
        "N_MONTHS": 6,
        "w_beta_H": 0.6,
        "w_alpha_H": 0.4,
        "risk_metrics": ["ShortfallProb"],
    }


@pytest.fixture
def sample_index_returns_basic() -> pd.Series:
    """Basic sample index returns for standard testing."""
    return pd.Series([0.01, 0.02, 0.015, 0.03, 0.005, 0.025] * 20)


@pytest.fixture
def sample_index_returns_mixed() -> pd.Series:
    """Mixed positive/negative returns for comprehensive testing."""
    return pd.Series([0.01, -0.02, 0.015, 0.005] * 3)


@pytest.fixture
def sample_index_returns_short() -> pd.Series:
    """Shorter series for quick reproducibility tests."""
    return pd.Series([0.01, 0.02, 0.015, 0.03, 0.005, 0.025])


# Tests for normal/valid scenarios

def test_orchestrator_runs_with_valid_config(valid_config_basic, sample_index_returns_basic) -> None:
    """Test that orchestrator runs successfully with valid configuration and returns expected outputs."""
    cfg = load_config(valid_config_basic)
    orch = SimulatorOrchestrator(cfg, sample_index_returns_basic)
    returns, summary = orch.run(seed=0)
    
    assert "Base" in returns
    assert "AnnReturn" in summary.columns


def test_orchestrator_reproducible_seed(valid_config_reproducibility, sample_index_returns_short) -> None:
    """Test that orchestrator produces identical results with same seed."""
    cfg = load_config(valid_config_reproducibility)
    orch = SimulatorOrchestrator(cfg, sample_index_returns_short)
    
    ret1, sum1 = orch.run(seed=42)
    ret2, sum2 = orch.run(seed=42)
    
    for key in ret1:
        if isinstance(ret1[key], pd.DataFrame):
            pd.testing.assert_frame_equal(ret1[key], ret2[key])
        else:
            np.testing.assert_allclose(ret1[key], ret2[key])
    pd.testing.assert_frame_equal(sum1, sum2)


def test_te_zero_when_no_alpha_allocation(valid_config_no_alpha, sample_index_returns_mixed) -> None:
    """Test that tracking error is zero when no alpha allocation is configured."""
    cfg = load_config(valid_config_no_alpha)
    orch = SimulatorOrchestrator(cfg, sample_index_returns_mixed)
    returns, summary = orch.run(seed=0)
    
    base = returns["Base"]
    internal_beta = returns["InternalBeta"]
    np.testing.assert_allclose(base, internal_beta)
    
    te_val = summary.loc[summary.Agent == "InternalBeta", "TE"].iloc[0]
    assert te_val == pytest.approx(0.0, abs=TEST_TOLERANCE_EPSILON)


# Tests for problematic/invalid scenarios

@pytest.mark.parametrize("w_beta_H,w_alpha_H,expected_error", [
    (0.7, 0.4, "must sum to 1"),  # Sum > 1.0
    (0.3, 0.5, "must sum to 1"),  # Sum < 1.0  
    (1.1, 0.0, "must be between 0 and 1"),  # beta > 1.0
    (0.0, 1.1, "must be between 0 and 1"),  # alpha > 1.0
])
def test_invalid_share_configurations_raise_error(w_beta_H, w_alpha_H, expected_error) -> None:
    """Test that invalid beta/alpha share combinations raise ValueError."""
    invalid_config = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "w_beta_H": w_beta_H,
        "w_alpha_H": w_alpha_H,
        "risk_metrics": ["ShortfallProb"],
    }
    
    with pytest.raises(ValueError, match=expected_error):
        load_config(invalid_config)
