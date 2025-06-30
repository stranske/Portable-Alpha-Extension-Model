from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as npt
from numpy.typing import NDArray

Array = NDArray[npt.float64]

@dataclass
class AgentParams:
    name: str
    capital_mm: float
    beta_share: float
    alpha_share: float
    extra_args: Dict[str, Any] | None = None

class Agent:
    """Abstract sleeve. Child classes implement ``monthly_returns``."""

    def __init__(self, p: AgentParams) -> None:
        self.p = p
        self.extra = p.extra_args or {}

    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        raise NotImplementedError
