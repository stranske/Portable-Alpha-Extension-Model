from __future__ import annotations

from dashboard.utils import normalize_share


def test_normalize_share_handles_percent_values() -> None:
    assert normalize_share(60.0) == 0.6


def test_normalize_share_preserves_fraction_values() -> None:
    assert normalize_share(0.6) == 0.6


def test_normalize_share_handles_none() -> None:
    assert normalize_share(None) is None
