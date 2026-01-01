"""Facade types for CLI and programmatic run entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import numpy as np
    import pandas as pd

    from .config import ModelConfig


@dataclass(slots=True)
class RunArtifacts:
    """Standardized outputs from a single simulation run."""

    config: "ModelConfig"
    index_series: "pd.Series"
    returns: dict[str, "np.ndarray"]
    summary: "pd.DataFrame"
    inputs: dict[str, Any]
    raw_returns: dict[str, "pd.DataFrame"]
    stress_delta: "pd.DataFrame | None" = None
    base_summary: "pd.DataFrame | None" = None
    manifest: Mapping[str, Any] | None = None
