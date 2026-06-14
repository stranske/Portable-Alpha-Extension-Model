"""Per-sleeve management & performance fee layer (issue #1904).

The core engine models financing/margin cost but historically produced only
*gross*-of-fee returns. For portable-alpha structures, external-manager
management fees and performance fees (on returns above a hurdle) are
first-order and frequently decide whether the structure beats plain beta.
Without them every output overstates the structure.

This module adds an opt-in, per-sleeve fee schedule so callers can produce
net-of-fee outputs alongside the existing gross behaviour. When no schedule is
supplied (the default) returns are unchanged and fully backward compatible.

Modelling choices (intentionally a simple first increment; see the PR for
rationale and follow-ups):

* Fees are deducted in **sleeve contribution-return space** -- directly from
  the monthly contribution return each agent emits
  (:meth:`pa_core.agents.types.Agent.monthly_returns`), which is already scaled
  by the sleeve's capital share. Assigning a schedule to a sleeve therefore
  charges fees on that sleeve's fund-level contribution, mirroring how
  financing cost is already deducted inside the agent return.
* **Management fee** (``mgmt_fee_bps``) is an annual rate in basis points,
  accrued linearly as a constant monthly drag ``mgmt_fee_bps / 1e4 / 12``.
* **Performance fee** (``perf_fee_pct``) is a fraction of the monthly return
  above an annual ``hurdle_bps`` hurdle (e.g. ``0.20`` for a 20% fee). It is
  crystallised monthly with no high-water mark and no rebate below the hurdle
  (negative excess is zero-clipped). A high-water-mark / notional-exact variant
  is a documented follow-up.

All fee parameters are non-negative; an all-zero schedule is a no-op.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .backend import xp as np
from .types import ArrayLike

__all__ = ["FeeSchedule", "compute_fee_drag", "apply_fees"]

_BPS_PER_UNIT = 10_000.0
_MONTHS_PER_YEAR = 12.0


class FeeSchedule(BaseModel):
    """Management + performance fee schedule for a single sleeve."""

    model_config = ConfigDict(frozen=True)

    mgmt_fee_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual management fee in basis points on notional.",
    )
    perf_fee_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Performance fee as a fraction of return above the hurdle (e.g. 0.20).",
    )
    hurdle_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual performance-fee hurdle in basis points.",
    )

    @property
    def is_zero(self) -> bool:
        """True when the schedule charges nothing (a no-op)."""
        return self.mgmt_fee_bps == 0.0 and self.perf_fee_pct == 0.0


def compute_fee_drag(gross: ArrayLike, schedule: FeeSchedule) -> ArrayLike:
    """Return the per-month fee drag matrix for ``gross`` contribution returns.

    The result has the same shape as ``gross`` and is always non-negative
    (fees never increase returns). ``net = gross - compute_fee_drag(gross, ...)``.
    """
    mgmt_monthly = schedule.mgmt_fee_bps / _BPS_PER_UNIT / _MONTHS_PER_YEAR
    drag = np.full_like(gross, mgmt_monthly)
    if schedule.perf_fee_pct > 0.0:
        hurdle_monthly = schedule.hurdle_bps / _BPS_PER_UNIT / _MONTHS_PER_YEAR
        excess = np.maximum(gross - hurdle_monthly, 0.0)
        drag = drag + schedule.perf_fee_pct * excess
    return drag


def apply_fees(gross: ArrayLike, schedule: FeeSchedule) -> ArrayLike:
    """Return ``gross`` net of the management + performance fees in ``schedule``."""
    if schedule.is_zero:
        return gross
    return gross - compute_fee_drag(gross, schedule)
