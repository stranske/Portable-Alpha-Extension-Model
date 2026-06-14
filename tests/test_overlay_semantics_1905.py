from __future__ import annotations

from pathlib import Path

from dashboard.glossary import tooltip
from pa_core.config import ModelConfig
from pa_core.portfolio import (
    OVERLAY_TOTAL_DESCRIPTION,
    compute_total_contribution_returns,
    is_base_only_config,
)


def test_total_tooltip_documents_base_exclusion() -> None:
    total_tooltip = tooltip("overlay total")

    assert total_tooltip == OVERLAY_TOTAL_DESCRIPTION
    assert "excludes Base" in total_tooltip


def test_primer_documents_overlay_total_semantics() -> None:
    primer = Path("docs/primer.md").read_text()

    assert "Total (overlay contribution)" in primer
    assert "excludes Base" in primer
    assert "Base-only run reports Total as zero" in primer


def test_base_only_config_is_flagged() -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")

    assert is_base_only_config(cfg) is True


def test_overlay_config_is_not_base_only() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        external_pa_capital=10.0,
    )

    assert is_base_only_config(cfg) is False


def test_total_contribution_excludes_base() -> None:
    import numpy as np

    base = np.array([[0.03, 0.04]])
    overlay = np.array([[0.01, -0.02]])

    total = compute_total_contribution_returns({"Base": base, "ExternalPA": overlay})

    assert total is not None
    np.testing.assert_allclose(total, overlay)
