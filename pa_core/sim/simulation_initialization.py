from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..random import spawn_agent_rngs_with_ids, spawn_rngs
from ..types import GeneratorLike

DEFAULT_FINANCING_AGENTS: tuple[str, ...] = ("internal", "external_pa", "active_ext")


@dataclass(slots=True)
class RunRNGBundle:
    """RNG bundle for a single simulation run."""

    rng_returns: GeneratorLike
    rng_regime: GeneratorLike
    rngs_financing: Mapping[str, GeneratorLike]
    substream_ids: Mapping[str, str]


@dataclass(slots=True)
class SweepRNGBundle:
    """RNG bundle for a parameter sweep run."""

    rng_returns: GeneratorLike
    rngs_financing: Mapping[str, GeneratorLike]
    substream_ids: Mapping[str, str]
    seed: int


def ensure_rng(seed: int | None, rng: GeneratorLike | None) -> GeneratorLike:
    """Return a generator, creating a deterministic one from ``seed`` if needed."""
    if rng is not None:
        return rng
    return spawn_rngs(seed, 1)[0]


def initialize_run_rngs(
    seed: int | None,
    *,
    financing_agents: Sequence[str] = DEFAULT_FINANCING_AGENTS,
    legacy_agent_rng: bool = False,
) -> RunRNGBundle:
    """Create per-run RNGs for returns, regimes, and financing."""
    base_rng = spawn_rngs(seed, 1)[0]
    child_seeds = base_rng.integers(0, 2**32, size=3, dtype="uint32")
    rng_returns = spawn_rngs(int(child_seeds[0]), 1)[0]
    rng_regime = spawn_rngs(int(child_seeds[1]), 1)[0]
    fin_seed = int(child_seeds[2])
    rngs_financing, substream_ids = spawn_agent_rngs_with_ids(
        fin_seed,
        financing_agents,
        legacy_order=legacy_agent_rng,
    )
    return RunRNGBundle(
        rng_returns=rng_returns,
        rng_regime=rng_regime,
        rngs_financing=rngs_financing,
        substream_ids=substream_ids,
    )


def initialize_sweep_rngs(
    seed: int | None,
    *,
    financing_agents: Sequence[str] = DEFAULT_FINANCING_AGENTS,
    legacy_agent_rng: bool = False,
) -> SweepRNGBundle:
    """Create per-sweep RNGs derived from the run seed."""
    base_rng = spawn_rngs(seed, 1)[0]
    child_seed = int(base_rng.integers(0, 2**32, dtype="uint32"))
    rng_returns = spawn_rngs(child_seed, 1)[0]
    rngs_financing, substream_ids = spawn_agent_rngs_with_ids(
        child_seed,
        financing_agents,
        legacy_order=legacy_agent_rng,
    )
    return SweepRNGBundle(
        rng_returns=rng_returns,
        rngs_financing=rngs_financing,
        substream_ids=substream_ids,
        seed=child_seed,
    )
