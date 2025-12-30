from __future__ import annotations

import pandas as pd

from pa_core.config import ModelConfig
from pa_core.reporting.attribution import (
    compute_sleeve_return_attribution,
    compute_sleeve_risk_attribution,
)


def test_compute_sleeve_return_attribution_basic() -> None:
    cfg = ModelConfig(
        **{
            "Number of simulations": 10,
            "Number of months": 12,
            "External PA capital (mm)": 100.0,
            "Active Extension capital (mm)": 100.0,
            "Internal PA capital (mm)": 50.0,
            "Total fund capital (mm)": 500.0,
            "In-House beta share": 0.6,
            "In-House alpha share": 0.4,
            "External PA alpha fraction": 0.6,
            "Active share (%)": 0.5,
            "In-House annual return (%)": 0.06,
            "In-House annual vol (%)": 0.02,
            "Alpha-Extension annual return (%)": 0.04,
            "Alpha-Extension annual vol (%)": 0.02,
            "External annual return (%)": 0.03,
            "External annual vol (%)": 0.02,
            "Internal financing mean (monthly %)": 0.001,
            "Internal financing vol (monthly %)": 0.0,
            "Internal monthly spike prob": 0.0,
            "Internal spike multiplier": 0.0,
            "External PA financing mean (monthly %)": 0.002,
            "External PA financing vol (monthly %)": 0.0,
            "External PA monthly spike prob": 0.0,
            "External PA spike multiplier": 0.0,
            "Active Ext financing mean (monthly %)": 0.0015,
            "Active Ext financing vol (monthly %)": 0.0,
            "Active Ext monthly spike prob": 0.0,
            "Active Ext spike multiplier": 0.0,
            "Analysis mode": "returns",
        }
    )
    # Synthetic monthly index series with 1% mean
    idx_series = pd.Series([0.01] * 120)

    df = compute_sleeve_return_attribution(cfg, idx_series)

    # Required columns should be present
    assert {"Agent", "Sub", "Return"} <= set(df.columns)
    assert not df.empty

    # Base sleeve components exist
    base = df[df["Agent"] == "Base"]
    assert {"Beta", "Alpha", "Financing"} <= set(base["Sub"].unique())

    # If capital was allocated, corresponding agent rows exist
    assert (df["Agent"] == "ExternalPA").any()
    assert (df["Agent"] == "ActiveExt").any()
    assert (df["Agent"] == "InternalPA").any()

    # Returns are numeric and finite
    assert pd.api.types.is_numeric_dtype(df["Return"])  # type: ignore[arg-type]
    assert df["Return"].abs().sum() >= 0.0


def test_compute_sleeve_risk_attribution_outputs_metrics() -> None:
    cfg = ModelConfig(
        **{
            "Number of simulations": 10,
            "Number of months": 12,
            "External PA capital (mm)": 100.0,
            "Active Extension capital (mm)": 100.0,
            "Internal PA capital (mm)": 50.0,
            "Total fund capital (mm)": 500.0,
            "In-House beta share": 0.6,
            "In-House alpha share": 0.4,
            "External PA alpha fraction": 0.6,
            "Active share (%)": 0.5,
            "In-House annual return (%)": 0.06,
            "In-House annual vol (%)": 0.02,
            "Alpha-Extension annual return (%)": 0.04,
            "Alpha-Extension annual vol (%)": 0.02,
            "External annual return (%)": 0.03,
            "External annual vol (%)": 0.02,
            "Internal financing mean (monthly %)": 0.001,
            "Internal financing vol (monthly %)": 0.0,
            "Internal monthly spike prob": 0.0,
            "Internal spike multiplier": 0.0,
            "External PA financing mean (monthly %)": 0.002,
            "External PA financing vol (monthly %)": 0.0,
            "External PA monthly spike prob": 0.0,
            "External PA spike multiplier": 0.0,
            "Active Ext financing mean (monthly %)": 0.0015,
            "Active Ext financing vol (monthly %)": 0.0,
            "Active Ext monthly spike prob": 0.0,
            "Active Ext spike multiplier": 0.0,
            "Analysis mode": "returns",
        }
    )
    idx_series = pd.Series([0.01, 0.02, 0.0, -0.01, 0.03])

    df = compute_sleeve_risk_attribution(cfg, idx_series)

    expected_cols = {
        "Agent",
        "BetaVol",
        "AlphaVol",
        "CorrWithIndex",
        "AnnVolApprox",
        "TEApprox",
    }
    assert expected_cols <= set(df.columns)
    assert not df.empty

    assert set(df["Agent"]) >= {
        "Base",
        "ExternalPA",
        "ActiveExt",
        "InternalPA",
        "InternalBeta",
    }
    assert df["CorrWithIndex"].between(-1.0, 1.0).all()
    assert (df[["BetaVol", "AlphaVol", "AnnVolApprox", "TEApprox"]] >= 0).all().all()
