import pandas as pd

from pa_core.reporting.constraints import build_constraint_report


def test_build_constraint_report_flags_breaches_and_driver() -> None:
    summary = pd.DataFrame(
        {
            "Agent": ["ExternalPA", "ActiveExt", "InternalPA", "Total"],
            "monthly_TE": [0.04, 0.01, 0.02, 0.03],
            "monthly_BreachProb": [0.02, 0.06, 0.01, 0.04],
            "monthly_CVaR": [0.02, 0.025, 0.01, 0.04],
        }
    )

    report = build_constraint_report(
        summary,
        max_te=0.02,
        max_breach=0.05,
        max_cvar=0.03,
    )

    assert not report.empty
    total_te = report[(report["Agent"] == "Total") & (report["Metric"] == "monthly_TE")]
    assert total_te["Driver"].iloc[0] == "ExternalPA"
    active_breach = report[
        (report["Agent"] == "ActiveExt") & (report["Metric"] == "monthly_BreachProb")
    ]
    assert active_breach["Driver"].iloc[0] == "ActiveExt"
