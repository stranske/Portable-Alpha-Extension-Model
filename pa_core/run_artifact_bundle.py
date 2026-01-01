from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class RunArtifact:
    """Portable snapshot describing a single run's reproducible outputs."""

    config: str
    index_hash: str
    seed: int | None
    manifest: Mapping[str, Any] | None
    outputs: Mapping[str, str]
