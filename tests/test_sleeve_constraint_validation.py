import pandas as pd
import pytest

from pa_core.config import ModelConfig, load_config
from pa_core.facade import RunOptions, run_single
from pa_core.reporting.constraints import validate_sleeve_constraints
from pa_core.sleeve_suggestor import SLEEVE_AGENTS


def _make_summary_df() -> pd.DataFrame:
    rows = [
        {
            "Agent": "Total",
            "monthly_TE": 0.04,
            "monthly_BreachProb": 0.2,
            "monthly_CVaR": 0.1,
            "terminal_ShortfallProb": 0.3,
        },
        {
            "Agent": "ExternalPA",
            "monthly_TE": 0.05,
            "monthly_BreachProb": 0.21,
            "monthly_CVaR": 0.11,
            "terminal_ShortfallProb": 0.31,
        },
        {
            "Agent": "ActiveExt",
            "monthly_TE": 0.06,
            "monthly_BreachProb": 0.22,
            "monthly_CVaR": 0.12,
            "terminal_ShortfallProb": 0.32,
        },
        {
            "Agent": "InternalPA",
            "monthly_TE": 0.07,
            "monthly_BreachProb": 0.23,
            "monthly_CVaR": 0.13,
            "terminal_ShortfallProb": 0.33,
        },
    ]
    return pd.DataFrame(rows)


def test_validate_sleeve_constraints_total_scope() -> None:
    summary_df = _make_summary_df()
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 1,
            "Number of months": 1,
            "financing_mode": "broadcast",
            "sleeve_max_te": 0.01,
            "sleeve_max_breach": 0.1,
            "sleeve_max_cvar": 0.02,
            "sleeve_max_shortfall": 0.15,
            "sleeve_constraint_scope": "total",
        }
    )

    violations = validate_sleeve_constraints(summary_df, cfg)

    assert violations
    assert all("Total" in violation for violation in violations)


def test_validate_sleeve_constraints_per_sleeve_scope() -> None:
    summary_df = _make_summary_df()
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 1,
            "Number of months": 1,
            "financing_mode": "broadcast",
            "sleeve_max_te": 0.01,
            "sleeve_max_breach": 0.1,
            "sleeve_max_cvar": 0.02,
            "sleeve_max_shortfall": 0.15,
            "sleeve_constraint_scope": "per_sleeve",
        }
    )

    violations = validate_sleeve_constraints(summary_df, cfg)

    assert violations
    assert not any("Total" in violation for violation in violations)
    assert all(any(sleeve in violation for sleeve in SLEEVE_AGENTS) for violation in violations)


def test_run_single_raises_on_sleeve_constraint_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3, "sleeve_validate_on_run": True}
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    called: dict[str, object] = {}

    def _fake_validate(summary_df: pd.DataFrame, cfg_arg: ModelConfig) -> list[str]:
        called["summary"] = summary_df
        called["cfg"] = cfg_arg
        return ["Total monthly_TE=0.5 exceeds Tracking error limit 0.1"]

    monkeypatch.setattr("pa_core.reporting.constraints.validate_sleeve_constraints", _fake_validate)

    with pytest.raises(ValueError, match="Sleeve constraint violations"):
        run_single(cfg, idx, RunOptions(seed=123))

    assert "summary" in called
    assert isinstance(called["summary"], pd.DataFrame)
    assert isinstance(called["cfg"], ModelConfig)
    assert called["cfg"].sleeve_validate_on_run is True
