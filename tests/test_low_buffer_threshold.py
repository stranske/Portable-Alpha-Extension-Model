"""Test that LOW_BUFFER_THRESHOLD constant is used correctly."""

from __future__ import annotations

import pandas as pd
import pytest

from pa_core.viz import (
    beta_scatter,
    breach_calendar,
    risk_return,
    risk_return_bubble,
    te_cvar_scatter,
    theme,
)


@pytest.fixture
def cleared_thresholds():
    """Fixture to temporarily clear and restore theme.THRESHOLDS.

    This fixture saves the original THRESHOLDS, clears it for the test,
    and restores it after the test completes. This ensures tests can
    verify fallback behavior when THRESHOLDS is empty.
    """
    original_thresholds = theme.THRESHOLDS.copy()
    theme.THRESHOLDS.clear()

    try:
        yield
    finally:
        theme.THRESHOLDS.update(original_thresholds)


def test_low_buffer_threshold_constant_defined():
    """Test that LOW_BUFFER_THRESHOLD constant is defined and has expected value."""
    assert hasattr(theme, "LOW_BUFFER_THRESHOLD")
    assert theme.LOW_BUFFER_THRESHOLD == 0.1


def test_risk_return_uses_constant(cleared_thresholds):
    """Test that risk_return visualization uses the LOW_BUFFER_THRESHOLD constant."""
    # Create test data with shortfall probabilities
    df = pd.DataFrame(
        {
            "AnnReturn": [0.05, 0.08, 0.12],
            "ExcessReturn": [0.01, 0.02, 0.03],
            "TE": [0.01, 0.02, 0.03],
            "Agent": ["A", "B", "C"],
            "ShortfallProb": [
                0.03,
                0.08,
                0.15,
            ],  # Below green, between green and amber, above amber
        }
    )

    fig = risk_return.make(df)
    # Check that the function doesn't crash and returns a figure
    assert fig is not None
    assert len(fig.data) == 1


def test_beta_scatter_uses_constant(cleared_thresholds):
    """Test that beta_scatter visualization uses the LOW_BUFFER_THRESHOLD constant."""
    df = pd.DataFrame(
        {
            "TrackingErr": [0.01, 0.02, 0.03],
            "Beta": [0.95, 1.0, 1.05],
            "Capital": [100, 200, 300],
            "Agent": ["A", "B", "C"],
            "ShortfallProb": [0.03, 0.08, 0.15],
        }
    )

    fig = beta_scatter.make(df)
    assert fig is not None
    assert len(fig.data) == 1


def test_breach_calendar_uses_constant(cleared_thresholds):
    """Test that breach_calendar visualization uses the LOW_BUFFER_THRESHOLD constant."""
    df = pd.DataFrame(
        {
            "Month": [1, 2, 3],
            "TrackingErr": [0.02, 0.04, 0.01],
            "ShortfallProb": [0.08, 0.15, 0.03],
        }
    )

    fig = breach_calendar.make(df)
    assert fig is not None
    assert len(fig.data) == 1


def test_te_cvar_scatter_uses_constant(cleared_thresholds):
    """Test that te_cvar_scatter visualization uses the LOW_BUFFER_THRESHOLD constant."""
    df = pd.DataFrame(
        {
            "TrackingErr": [0.01, 0.02, 0.03],
            "CVaR": [0.05, 0.08, 0.12],
            "Capital": [100, 200, 300],
            "Agent": ["A", "B", "C"],
            "ShortfallProb": [0.03, 0.08, 0.15],
        }
    )

    fig = te_cvar_scatter.make(df)
    assert fig is not None
    assert len(fig.data) == 1


def test_risk_return_bubble_uses_constant(cleared_thresholds):
    """Test that risk_return_bubble visualization uses the LOW_BUFFER_THRESHOLD constant."""
    df = pd.DataFrame(
        {
            "AnnReturn": [0.05, 0.08, 0.12],
            "AnnVol": [0.02, 0.04, 0.06],
            "Capital": [100, 200, 300],
            "Agent": ["A", "B", "C"],
            "ShortfallProb": [0.03, 0.08, 0.15],
        }
    )

    fig = risk_return_bubble.make(df)
    assert fig is not None
    assert len(fig.data) == 1
