from __future__ import annotations
from .types import Agent, Array

class InternalBetaAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        return self.p.beta_share * (r_beta - financing)
