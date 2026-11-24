"""Tests for issue #2534 — regime disclaimer includes single-run regime switching."""

from pa_core.reporting.disclaimers import MODEL_LIMITATIONS


def test_limitations_mention_single_run_regimes() -> None:
    """The regime caveat in MODEL_LIMITATIONS must state that regimes apply to single runs."""
    regime_caveats = [item for item in MODEL_LIMITATIONS if "regime" in item.lower()]
    assert regime_caveats, "Expected a regime caveat in MODEL_LIMITATIONS"
    regime_caveat = regime_caveats[0].lower()
    assert "single runs" in regime_caveat
    assert "parameter sweeps" in regime_caveat
    assert "regimes apply in parameter sweeps" not in regime_caveat
