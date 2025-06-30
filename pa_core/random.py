from __future__ import annotations

from .backend import xp as np
from numpy.random import Generator, SeedSequence

__all__ = ["spawn_rngs"]


def spawn_rngs(seed: int, n: int) -> list[Generator]:
    """Return ``n`` independent generators derived from ``seed``."""
    if n <= 0:
        raise ValueError("n must be positive")
    ss = SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]

