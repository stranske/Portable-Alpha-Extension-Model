"""Internal-PA financing cost resolution.

The InternalPA sleeve funds an in-house alpha position. Holding that position
carries an internal financing cost that may be positive (a cost) or negative
(a benefit / positive carry). This module turns the scenario configuration
into a monthly financing-cost matrix that the InternalPA agent subtracts from
its sleeve return.

Unlike :func:`pa_core.sim.financing.simulate_financing`, the values produced
here are **never clipped** to be non-negative: negative financing costs are a
first-class, supported input (see issue #1849).

Resolution priority (highest first):

1. ``series`` -- an explicit per-month cost list (negatives allowed).
2. ``index`` -- a named built-in index futures/swap financing curve.
3. ``mean_month`` -- a deterministic monthly cost, optionally perturbed by a
   ``sigma_month`` stochastic draw.

When none of these express a non-zero cost (the defaults), a zero matrix is
returned so that runs with internal-PA financing disabled are bit-for-bit
identical to historical behaviour.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import numpy.typing as npt

from ..backend import xp as np
from ..random import spawn_rngs
from ..types import GeneratorLike
from ..data.index_financing_curves import get_index_financing_curve_monthly

_FINANCING_MODES = ("broadcast", "per_path")


def resolve_internal_pa_financing_series(
    *,
    n_months: int,
    n_sim: int,
    mean_month: float = 0.0,
    sigma_month: float = 0.0,
    series: Optional[Sequence[float]] = None,
    index: Optional[str] = None,
    financing_mode: str = "broadcast",
    rng: Optional[GeneratorLike] = None,
) -> npt.NDArray[Any]:
    """Return an ``(n_sim, n_months)`` internal-PA monthly financing matrix.

    Negative costs are preserved. See module docstring for resolution order.
    """
    if n_months <= 0:
        raise ValueError("n_months must be positive")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if financing_mode not in _FINANCING_MODES:
        raise ValueError(f"financing_mode must be one of: {sorted(_FINANCING_MODES)}")

    # 1. Explicit per-month series.
    if series is not None:
        values = [float(v) for v in series]
        if len(values) != n_months:
            raise ValueError(
                "internal_pa_financing_series length "
                f"({len(values)}) must equal n_months ({n_months})"
            )
        row = np.asarray(values, dtype=float)
        return cast(npt.NDArray[Any], np.broadcast_to(row, (n_sim, n_months)).copy())

    # 2. Named index futures/swap financing curve.
    if index is not None:
        curve = get_index_financing_curve_monthly(index, n_months)
        row = np.asarray(curve, dtype=float)
        return cast(npt.NDArray[Any], np.broadcast_to(row, (n_sim, n_months)).copy())

    # 3. Deterministic mean, optionally perturbed by a stochastic draw.
    if sigma_month and sigma_month > 0.0:
        if rng is None:
            rng = spawn_rngs(None, 1)[0]
        assert rng is not None
        n_scenarios = 1 if financing_mode == "broadcast" else n_sim
        draw = rng.normal(loc=mean_month, scale=sigma_month, size=(n_scenarios, n_months))
        # No clip: negative financing costs are a supported input.
        if n_scenarios == 1:
            return cast(npt.NDArray[Any], np.broadcast_to(draw[0], (n_sim, n_months)).copy())
        return cast(npt.NDArray[Any], draw)

    # Deterministic constant (includes the all-defaults zero case).
    return cast(npt.NDArray[Any], np.full((n_sim, n_months), float(mean_month), dtype=float))
