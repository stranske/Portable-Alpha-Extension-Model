"""Synthetic index futures/swap financing-cost curves.

These curves provide a non-proprietary, deterministic fallback for the
internal-PA financing cost of a position held via a major index future or
swap. Values are illustrative annualised financing spreads expressed in
basis points (bps) over the relevant funding rate; they are *not* observed
market data and must not be treated as a calibration source for production
risk numbers.

Sign convention
---------------
A *positive* value is a financing **cost** (it lowers the InternalPA sleeve
return); a *negative* value is a financing **benefit / positive carry** (it
raises the sleeve return). Negative values are deliberately supported and are
never clipped -- see ``stranske/Portable-Alpha-Extension-Model`` issue #1849.

Typical historical ranges for index financing spreads sit in the ~20-120 bps
band, but those ranges are calibration context only and are never enforced as
hard limits.
"""

from __future__ import annotations

from typing import Dict, List

MONTHS_PER_YEAR = 12

# Synthetic annualised financing spreads (bps) for major index futures/swaps.
# Positive = cost, negative = benefit/positive carry. Illustrative only.
INDEX_FINANCING_CURVES_BPS: Dict[str, float] = {
    "SPX": 35.0,  # S&P 500
    "NDX": 45.0,  # Nasdaq-100
    "RTY": 60.0,  # Russell 2000
    "SX5E": 25.0,  # Euro Stoxx 50
    "UKX": 30.0,  # FTSE 100
    "NKY": -15.0,  # Nikkei 225 -- illustrative positive-carry / negative cost
    "TPX": -10.0,  # TOPIX -- illustrative positive-carry / negative cost
}


def available_indices() -> List[str]:
    """Return the sorted list of indices with a built-in financing curve."""
    return sorted(INDEX_FINANCING_CURVES_BPS)


def annual_bps_to_monthly(bps: float) -> float:
    """Convert an annualised financing spread in bps to a monthly return cost.

    100 bps annual == 0.01 annual == 0.01 / 12 monthly. Linear (simple)
    conversion is used so positive and negative spreads stay symmetric.
    """
    return float(bps) / 10_000.0 / MONTHS_PER_YEAR


def get_index_financing_curve_monthly(index: str, n_months: int) -> List[float]:
    """Return an ``n_months``-long monthly internal-PA financing cost series.

    The synthetic curves are flat over time (a single representative spread is
    broadcast across the horizon). Negative spreads are preserved unchanged.

    Raises
    ------
    KeyError
        If ``index`` is not a known curve. Use :func:`available_indices`.
    ValueError
        If ``n_months`` is not positive.
    """
    if n_months <= 0:
        raise ValueError("n_months must be positive")
    key = index.strip().upper()
    if key not in INDEX_FINANCING_CURVES_BPS:
        raise KeyError(
            f"Unknown index financing curve {index!r}; " f"available: {available_indices()}"
        )
    monthly_cost = annual_bps_to_monthly(INDEX_FINANCING_CURVES_BPS[key])
    return [monthly_cost] * int(n_months)
