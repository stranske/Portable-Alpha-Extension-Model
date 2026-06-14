import pandas as pd
import pytest

from pa_core.sweep import aggregate_sweep_results


def test_aggregate_sweep_results_computes_stats() -> None:
    results = [
        {
            "combination_id": 0,
            "parameters": {"mu_H": 0.01},
            "summary": pd.DataFrame(
                {
                    "Agent": ["Base"],
                    "terminal_AnnReturn": [0.1],
                    "terminal_AnnReturn_P50": [0.08],
                    "terminal_AnnReturn_P10": [0.02],
                    "terminal_AnnReturn_P90": [0.12],
                    "monthly_AnnVol": [0.2],
                    "monthly_TE": [0.3],
                    "monthly_TE_PerPath": [0.25],
                }
            ),
        },
        {
            "combination_id": 1,
            "parameters": {"mu_H": 0.02},
            "summary": pd.DataFrame(
                {
                    "Agent": ["Base"],
                    "terminal_AnnReturn": [0.2],
                    "terminal_AnnReturn_P50": [0.16],
                    "terminal_AnnReturn_P10": [0.04],
                    "terminal_AnnReturn_P90": [0.24],
                    "monthly_AnnVol": [0.4],
                    "monthly_TE": [0.5],
                    "monthly_TE_PerPath": [0.35],
                }
            ),
        },
    ]

    stats = aggregate_sweep_results(results, percentiles=(0.5,))

    ann_row = stats[(stats["Agent"] == "Base") & (stats["Metric"] == "terminal_AnnReturn")].iloc[0]
    assert ann_row["Mean"] == pytest.approx(0.15)
    assert ann_row["Std"] == pytest.approx(0.070710678, rel=1e-6)
    assert ann_row["P50"] == pytest.approx(0.15)

    p50_row = stats[
        (stats["Agent"] == "Base") & (stats["Metric"] == "terminal_AnnReturn_P50")
    ].iloc[0]
    assert p50_row["Mean"] == pytest.approx(0.12)

    per_path_row = stats[
        (stats["Agent"] == "Base") & (stats["Metric"] == "monthly_TE_PerPath")
    ].iloc[0]
    assert per_path_row["Mean"] == pytest.approx(0.30)
