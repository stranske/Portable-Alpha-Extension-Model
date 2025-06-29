from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

Array = np.ndarray

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


class BaseAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape or r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        return (
            self.p.beta_share * (r_beta - financing)
            + self.p.alpha_share * alpha_stream
        )


class ExternalPAAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape or r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        theta = float(self.extra.get("theta_extpa", 0.0))
        return (
            self.p.beta_share * (r_beta - financing)
            + (self.p.beta_share * theta) * alpha_stream
        )


class ActiveExtensionAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape or r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        active_share = float(self.extra.get("active_share", 0.5))
        return (
            self.p.beta_share * (r_beta - financing)
            + (self.p.beta_share * active_share) * alpha_stream
        )


class InternalBetaAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        return self.p.beta_share * (r_beta - financing)


class InternalPAAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape:
            raise ValueError("shape mismatch")
        return self.p.alpha_share * alpha_stream

