from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Explicitly declare as a type alias for Pyright
Array: TypeAlias = NDArray[np.float64]


@dataclass
class AgentParams:
    name: str
    capital_mm: float
    beta_share: float
    alpha_share: float
    extra_args: dict[str, Any] | None = None


class Agent:
    """Abstract sleeve. Child classes implement ``monthly_returns``."""

    def __init__(self, p: AgentParams) -> None:
        self.p = p
        self.extra = p.extra_args or {}

    def _validate_inputs(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> None:
        """Validate that all arrays are 2D and share the same shape."""
        if r_beta.ndim != 2 or alpha_stream.ndim != 2 or financing.ndim != 2:
            raise ValueError("inputs must be 2D (n_sim, n_months)")
        if r_beta.shape != alpha_stream.shape or r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")

    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        raise NotImplementedError
