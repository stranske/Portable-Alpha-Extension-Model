"""Tier 0 golden master: per-agent metrics of the funded baseline run.

Re-bless after an intended change:
    pytest tests/baseline/test_golden.py --force-regen
then review the diff and commit.
"""

from __future__ import annotations

from baseline_kit import check_metrics

from . import adapter


def test_baseline_summary_golden(baseline_summary, num_regression):
    check_metrics(num_regression, adapter.flat_metrics(baseline_summary))
