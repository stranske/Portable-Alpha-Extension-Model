"""Tier 3 economic invariants across all agents on the baseline run."""

from __future__ import annotations

from baseline_kit import assert_invariants

from . import invariants


def test_baseline_invariants(baseline_summary):
    assert_invariants(invariants.check_run(baseline_summary), context="PAEM baseline")
