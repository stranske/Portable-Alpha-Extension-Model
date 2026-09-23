"""Tests for issue #1923 — disclose model limitations in docs & board-pack.

Verifies the single source of truth (``pa_core.reporting.disclaimers``) is wired
into both the README and the generated PPTX board pack so the two cannot drift.
"""

import tempfile
from pathlib import Path

import pandas as pd
from pptx import Presentation

from pa_core.reporting.disclaimers import (
    LIMITATIONS_TITLE,
    MODEL_LIMITATIONS,
    limitations_markdown,
)
from pa_core.reporting.export_packet import create_export_packet

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_limitations_cover_required_caveats():
    """Every caveat named in the audit issue must be present."""
    blob = " ".join(MODEL_LIMITATIONS).lower()
    for phrase in (
        "gross of fees",
        "total excludes base",
        "i.i.d",
        "regimes apply in parameter sweeps",
        "broadcast",
        "not been backtested",
        "risk_metrics",
        "scenario.sleeves",
        "floored at zero",
        "positive carry",
        "single_with_sensitivity",
        "sweep engine",
    ):
        assert phrase in blob, f"missing caveat: {phrase}"
    assert "regimes are ignored" not in blob, "stale caveat: sweeps honor regimes"


def test_limitations_markdown_is_a_bullet_list():
    md = limitations_markdown()
    assert md.count("- ") == len(MODEL_LIMITATIONS)
    assert MODEL_LIMITATIONS[0] in md


def test_readme_documents_limitations():
    readme = (_REPO_ROOT / "README.md").read_text()
    section_start = readme.index(f"## {LIMITATIONS_TITLE}")
    next_section = readme.index("\n## ", section_start + 1)
    section = readme[section_start:next_section]
    assert limitations_markdown() in section


def test_parameter_guide_documents_advisory_and_unwired_fields():
    guide = (_REPO_ROOT / "docs/guides/PARAMETER_GUIDE.md").read_text()
    for phrase in (
        "`risk_metrics` controls which metrics are reported",
        "`Scenario.sleeves` is validated by the scenario schema",
        "legacy financing costs",
        "floored at zero",
        "`analysis_mode: single_with_sensitivity`",
        "parameter sweep engine supports `returns`, `capital`,",
    ):
        assert phrase in guide


def test_board_pack_includes_limitations_slide():
    """The generated PPTX must carry a limitations slide with every caveat."""
    summary_df = pd.DataFrame(
        {
            "terminal_AnnReturn": [0.05],
            "monthly_AnnVol": [0.12],
            "terminal_ShortfallProb": [0.1],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pptx_path, _ = create_export_packet(
            figs=[],
            summary_df=summary_df,
            raw_returns_dict={"Summary": summary_df},
            inputs_dict={"N_SIMULATIONS": 10, "N_MONTHS": 12},
            base_filename=str(Path(tmpdir) / "packet"),
        )
        prs = Presentation(pptx_path)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text_frame.text)
        blob = "\n".join(texts)
        assert LIMITATIONS_TITLE in blob
        for item in MODEL_LIMITATIONS:
            assert item in blob, f"slide missing caveat: {item}"


def _sweep_total_ann_return(cfg, idx):
    from pa_core import sweep as sweep_module
    from pa_core.contracts import SUMMARY_AGENT_COLUMN, SUMMARY_ANN_RETURN_COLUMN
    from pa_core.random import spawn_agent_rngs, spawn_rngs

    rng_returns = spawn_rngs(42, 1)[0]
    fin_rngs = spawn_agent_rngs(42, ["internal", "external_pa", "active_ext"])
    summary = sweep_module.run_parameter_sweep(cfg, idx, rng_returns, fin_rngs, seed=42)[0][
        "summary"
    ]
    total = summary[summary[SUMMARY_AGENT_COLUMN] == "Total"].iloc[0]
    return float(total[SUMMARY_ANN_RETURN_COLUMN])


def test_regime_sweep_changes_metrics_when_configured():
    """Sweeps honor configured regimes, and the published caveats must say so."""
    from pa_core.config import RegimeConfig, load_config

    idx = pd.Series([0.01, -0.02, 0.015] * 4)
    base = load_config(_REPO_ROOT / "examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 400, "N_MONTHS": 12, "analysis_mode": "returns"}
    )
    with_regimes = base.model_copy(
        update={
            "regimes": [
                RegimeConfig(name="calm"),
                RegimeConfig(name="stress", idx_sigma_multiplier=3.0),
            ],
            "regime_transition": [[0.5, 0.5], [0.5, 0.5]],
            "regime_start": "calm",
        }
    )

    plain = _sweep_total_ann_return(base, idx)
    regime = _sweep_total_ann_return(with_regimes, idx)
    assert abs(regime - plain) > 1e-4

    # Behaviour and disclosure must agree: regimes move sweep metrics, so no
    # caveat may claim sweeps ignore them.
    blob = " ".join(MODEL_LIMITATIONS).lower()
    assert "regimes are ignored" not in blob
    assert "regimes apply in parameter sweeps" in blob
