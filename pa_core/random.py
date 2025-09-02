from __future__ import annotations

from typing import Any, Dict, List

from .backend import xp

try:  # pragma: no cover - numpy is always available for typing
    from numpy.random import Generator  # type: ignore[reportMissingTypeStubs]
except Exception:  # pragma: no cover
    Generator = Any  # fall back for type checking

__all__ = ["spawn_rngs", "spawn_agent_rngs"]


def spawn_rngs(seed: int | None, n: int) -> List[Any]:
    """Return ``n`` independent generators derived from ``seed``.

    Passing ``None`` uses unpredictable entropy from the OS.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ss = xp.random.SeedSequence(seed)
    return [xp.random.default_rng(s) for s in ss.spawn(n)]


def spawn_agent_rngs(seed: int | None, agent_names: List[str]) -> Dict[str, Any]:
    """Return a dedicated RNG for each agent name derived from ``seed``."""
    if not agent_names:
        raise ValueError("agent_names must not be empty")
    ss = xp.random.SeedSequence(seed)
    spawned = ss.spawn(len(agent_names))
    rngs = [xp.random.default_rng(s) for s in spawned]
    return dict(zip(agent_names, rngs))
