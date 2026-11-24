"""Tests for issue #2534 — regime disclaimer includes single-run regime switching."""

from pathlib import Path

import pandas as pd

from pa_core.config import RegimeConfig, load_config
from pa_core.contracts import SUMMARY_AGENT_COLUMN, SUMMARY_ANN_RETURN_COLUMN
from pa_core.facade import RunOptions, run_single
from pa_core.reporting.disclaimers import MODEL_LIMITATIONS

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_limitations_mention_single_run_regimes() -> None:
    """The regime caveat in MODEL_LIMITATIONS must state that regimes apply to single runs."""
    regime_caveats = [item for item in MODEL_LIMITATIONS if "regime" in item.lower()]
    assert len(regime_caveats) == 1, "Expected a single regime caveat in MODEL_LIMITATIONS"
    regime_caveat = regime_caveats[0].lower()
    assert "single runs" in regime_caveat
    assert "parameter sweeps" in regime_caveat
    assert "regimes apply in parameter sweeps" not in regime_caveat


def test_single_run_regimes_change_metrics_matching_disclaimer() -> None:
    """Single runs honor configured regimes, matching the published disclaimer."""
    idx = pd.Series(
        [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.01, 0.005, -0.015, 0.02, 0.01, -0.005]
    )
    base = load_config(_REPO_ROOT / "examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 200, "N_MONTHS": 12, "analysis_mode": "capital"}
    )
    with_regimes = base.model_copy(
        update={
            "regimes": [
                RegimeConfig(name="calm"),
                RegimeConfig(name="stress", idx_sigma_multiplier=2.5),
            ],
            "regime_transition": [[0.9, 0.1], [0.1, 0.9]],
            "regime_start": "calm",
        }
    )

    def _total_ann_return(cfg) -> float:
        artifacts = run_single(cfg, idx, RunOptions(seed=99))
        summary = artifacts.summary
        total = summary[summary[SUMMARY_AGENT_COLUMN] == "Total"].iloc[0]
        return float(total[SUMMARY_ANN_RETURN_COLUMN])

    plain = _total_ann_return(base)
    regime = _total_ann_return(with_regimes)
    assert abs(regime - plain) > 1e-4

    blob = " ".join(MODEL_LIMITATIONS).lower()
    assert "single runs" in blob
    assert "regimes apply to single runs and parameter sweeps" in blob
