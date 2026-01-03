from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, cast

import numpy.typing as npt

from ..backend import xp as np
from ..random import spawn_rngs
from ..types import GeneratorLike

logger = logging.getLogger(__name__)

_FINANCING_MODES = ("broadcast", "per_path")
_BROADCAST_WARN_THRESHOLD = 100


def simulate_financing(
    T: int,
    financing_mean: float,
    financing_sigma: float,
    spike_prob: float,
    spike_factor: float,
    *,
    seed: Optional[int] = None,
    n_scenarios: int = 1,
    rng: Optional[GeneratorLike] = None,
) -> npt.NDArray[Any]:
    """Vectorised financing spread simulation with optional spikes."""
    if T <= 0:
        raise ValueError("T must be positive")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    if rng is None:
        rng = spawn_rngs(seed, 1)[0]
    assert rng is not None
    base = rng.normal(loc=financing_mean, scale=financing_sigma, size=(n_scenarios, T))
    jumps = (rng.random(size=(n_scenarios, T)) < spike_prob) * (spike_factor * financing_sigma)
    out = cast(npt.NDArray[Any], np.clip(base + jumps, 0.0, None))
    if n_scenarios == 1:
        return cast(npt.NDArray[Any], out[0])
    return out


def draw_financing_series(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    financing_mode: str,
    rng: Optional[GeneratorLike] = None,
    rngs: Optional[Mapping[str, GeneratorLike]] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Return three matrices of monthly financing spreads.

    ``financing_mode`` controls whether a single monthly vector is broadcast
    across all scenarios ("broadcast") or each path receives its own draw
    ("per_path").

    ``rngs`` may provide dedicated generators for each sleeve under the keys
    ``"internal"``, ``"external_pa"``, and ``"active_ext"``. If not supplied,
    ``rng`` will be used for all sleeves.
    """
    if financing_mode not in _FINANCING_MODES:
        raise ValueError(f"financing_mode must be one of: {sorted(_FINANCING_MODES)}")
    if n_months <= 0:
        raise ValueError("n_months must be positive")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if financing_mode == "broadcast" and n_sim >= _BROADCAST_WARN_THRESHOLD:
        logger.warning(
            "Financing mode 'broadcast' reuses a single financing path across %s scenarios.",
            n_sim,
        )

    if rngs is not None:
        tmp_int = rngs.get("internal")
        r_int = tmp_int if tmp_int is not None else spawn_rngs(None, 1)[0]

        tmp_ext = rngs.get("external_pa")
        r_ext = tmp_ext if tmp_ext is not None else spawn_rngs(None, 1)[0]

        tmp_act = rngs.get("active_ext")
        r_act = tmp_act if tmp_act is not None else spawn_rngs(None, 1)[0]
    else:
        if rng is None:
            rng = spawn_rngs(None, 1)[0]
        r_int = rng
        r_ext = rng
        r_act = rng

    def _sim(
        mean_key: str,
        sigma_key: str,
        p_key: str,
        k_key: str,
        rng_local: GeneratorLike,
    ) -> npt.NDArray[Any]:
        mean = params[mean_key]
        sigma = params[sigma_key]
        p = params[p_key]
        k = params[k_key]
        draw = simulate_financing(
            n_months,
            mean,
            sigma,
            p,
            k,
            n_scenarios=1 if financing_mode == "broadcast" else n_sim,
            rng=rng_local,
        )
        if draw.ndim == 1:
            return cast(npt.NDArray[Any], np.broadcast_to(draw, (n_sim, n_months)))
        return draw

    f_int_mat = _sim(
        "internal_financing_mean_month",
        "internal_financing_sigma_month",
        "internal_spike_prob",
        "internal_spike_factor",
        r_int,
    )
    f_ext_pa_mat = _sim(
        "ext_pa_financing_mean_month",
        "ext_pa_financing_sigma_month",
        "ext_pa_spike_prob",
        "ext_pa_spike_factor",
        r_ext,
    )
    f_act_ext_mat = _sim(
        "act_ext_financing_mean_month",
        "act_ext_financing_sigma_month",
        "act_ext_spike_prob",
        "act_ext_spike_factor",
        r_act,
    )
    return f_int_mat, f_ext_pa_mat, f_act_ext_mat
