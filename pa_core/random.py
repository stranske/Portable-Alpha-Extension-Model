from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Dict, List, Sequence

from .backend import xp
from .types import GeneratorLike

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "spawn_rngs",
    "spawn_agent_rngs",
    "spawn_agent_rngs_with_ids",
    "derive_agent_substream_ids",
]


def spawn_rngs(seed: int | None, n: int) -> List[GeneratorLike]:
    """Return ``n`` independent generators derived from ``seed``.

    Passing ``None`` uses unpredictable entropy from the OS.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ss = xp.random.SeedSequence(seed)
    return [xp.random.default_rng(s) for s in ss.spawn(n)]


def spawn_agent_rngs(
    seed: int | None, agent_names: List[str], *, legacy_order: bool = False
) -> Dict[str, GeneratorLike]:
    """Return a dedicated RNG for each agent name derived from ``seed``.

    When ``legacy_order`` is False (default), substreams are derived from a
    stable hash of the seed and agent name, with names sorted before assignment.
    """
    rngs, _ = _build_agent_substreams(seed, agent_names, legacy_order=legacy_order)
    return rngs


def spawn_agent_rngs_with_ids(
    seed: int | None, agent_names: Sequence[str], *, legacy_order: bool = False
) -> tuple[Dict[str, GeneratorLike], Dict[str, str]]:
    """Return RNGs plus the substream identifiers used to build them."""
    return _build_agent_substreams(seed, agent_names, legacy_order=legacy_order)


def derive_agent_substream_ids(
    seed: int | None, agent_names: Sequence[str], *, legacy_order: bool = False
) -> Dict[str, str]:
    """Return deterministic substream identifiers for each agent name."""
    _, substream_ids = _build_agent_substreams(seed, agent_names, legacy_order=legacy_order)
    return substream_ids


def _build_agent_substreams(
    seed: int | None, agent_names: Sequence[str], *, legacy_order: bool = False
) -> tuple[Dict[str, GeneratorLike], Dict[str, str]]:
    names = _normalize_agent_names(agent_names, legacy_order=legacy_order)
    if legacy_order:
        base = xp.random.SeedSequence(seed)
        spawned = base.spawn(len(names))
        rngs: Dict[str, GeneratorLike] = {}
        substream_ids: Dict[str, str] = {}
        for name, child in zip(names, spawned):
            rngs[name] = xp.random.default_rng(child)
            substream_ids[name] = _legacy_substream_id(child)
        return rngs, substream_ids

    base_entropy = _base_entropy(seed)
    seed_token = _seed_token(seed, base_entropy)
    rngs = {}
    substream_ids = {}
    for name in names:
        sub_id = _stable_substream_id(seed_token, name)
        rngs[name] = xp.random.default_rng(xp.random.SeedSequence(_entropy_from_id(sub_id)))
        substream_ids[name] = sub_id
    return rngs, substream_ids


def _normalize_agent_names(agent_names: Sequence[str], *, legacy_order: bool) -> list[str]:
    if not agent_names:
        raise ValueError("agent_names must not be empty")
    names = list(agent_names)
    if len(set(names)) != len(names):
        raise ValueError("agent_names must be unique")
    if not legacy_order:
        names = sorted(names)
    return names


def _base_entropy(seed: int | None) -> int:
    return int(xp.random.SeedSequence(seed).entropy)


def _seed_token(seed: int | None, base_entropy: int) -> str:
    if seed is None:
        return f"none|{base_entropy}"
    return str(seed)


def _stable_substream_id(seed_token: str, agent_name: str) -> str:
    token = f"pa-core-rng-v1|{seed_token}|{agent_name}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _legacy_substream_id(seed_sequence: "np.random.SeedSequence") -> str:
    token = f"legacy|{seed_sequence.entropy}|{seed_sequence.spawn_key}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _entropy_from_id(substream_id: str) -> int:
    return int(substream_id[:32], 16)
