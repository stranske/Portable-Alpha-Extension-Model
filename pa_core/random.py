from __future__ import annotations

from numpy.random import Generator, SeedSequence

from .backend import xp as np

__all__ = ["spawn_rngs", "spawn_agent_rngs"]


def spawn_rngs(seed: int | None, n: int) -> list[Generator]:
    """Return ``n`` independent generators derived from ``seed``.

    Passing ``None`` uses unpredictable entropy from the OS.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ss = SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def spawn_agent_rngs(seed: int | None, agent_names: list[str]) -> dict[str, Generator]:
    """Return a dedicated RNG for each agent name derived from ``seed``."""
    if not agent_names:
        raise ValueError("agent_names must not be empty")
    ss = SeedSequence(seed)
    spawned = ss.spawn(len(agent_names))
    rngs = [np.random.default_rng(s) for s in spawned]
    return dict(zip(agent_names, rngs))
