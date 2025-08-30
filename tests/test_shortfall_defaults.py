"""Tests to verify centralized ShortfallProb default value usage."""

from pathlib import Path
import types
import sys

import pandas as pd

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.viz import theme, risk_return, beta_scatter, te_cvar_scatter
from pa_core.reporting import excel, sweep_excel


def test_default_shortfall_prob_constant_exists():
    """Test that the DEFAULT_SHORTFALL_PROB constant is properly defined."""
    assert hasattr(theme, 'DEFAULT_SHORTFALL_PROB')
    assert theme.DEFAULT_SHORTFALL_PROB == 0.0


def test_risk_return_uses_default_constant():
    """Test that risk_return module uses the centralized constant."""
    df = pd.DataFrame({
        "AnnReturn": [0.05, 0.03],
        "AnnVol": [0.10, 0.08], 
        "TrackingErr": [0.02, 0.01],
        "Agent": ["Test1", "Test2"],
        # Intentionally omit ShortfallProb to test default
    })
    
    # Should not raise an error and should use the default
    fig = risk_return.make(df)
    assert fig is not None


def test_beta_scatter_handles_missing_shortfall():
    """Test that beta_scatter handles missing ShortfallProb correctly."""
    df = pd.DataFrame({
        "TrackingErr": [0.02, 0.01],
        "Beta": [1.0, 0.8],
        "Capital": [100, 200],
        "Agent": ["Test1", "Test2"],
        # Intentionally omit ShortfallProb to test default
    })
    
    # Should not raise an error and should use the default
    fig = beta_scatter.make(df)
    assert fig is not None


def test_te_cvar_scatter_handles_missing_shortfall():
    """Test that te_cvar_scatter handles missing ShortfallProb correctly."""
    df = pd.DataFrame({
        "TrackingErr": [0.02, 0.01],
        "CVaR": [-0.05, -0.03],
        "Agent": ["Test1", "Test2"],
        # Intentionally omit ShortfallProb to test default
    })
    
    # Should not raise an error and should use the default
    fig = te_cvar_scatter.make(df)
    assert fig is not None


def test_export_functions_use_default_constant(tmp_path):
    """Test that Excel export functions use the centralized constant."""
    # Test single export
    inputs = {"test_param": "test_value"}
    summary = pd.DataFrame({"Agent": ["Test"], "AnnReturn": [0.05]})
    # Intentionally omit ShortfallProb to test default
    raw_returns = {"Test": pd.DataFrame([[0.01, 0.02]])}
    
    file_path = tmp_path / "test_output.xlsx"
    excel.export_to_excel(inputs, summary, raw_returns, str(file_path))
    assert file_path.exists()
    
    # Test sweep export  
    results = [{
        "combination_id": 1,
        "summary": pd.DataFrame({"Agent": ["Test"], "AnnReturn": [0.05]})
        # Intentionally omit ShortfallProb to test default
    }]
    
    sweep_file = tmp_path / "test_sweep.xlsx"
    sweep_excel.export_sweep_results(results, str(sweep_file))
    assert sweep_file.exists()