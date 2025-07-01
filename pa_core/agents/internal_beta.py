from __future__ import annotations
from .types import Agent, Array

class InternalBetaAgent(Agent):
    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        """Return margin sleeve returns with input validation."""
        if r_beta.shape != financing.shape or r_beta.shape != alpha_stream.shape:
            raise ValueError("shape mismatch")
        return self.p.beta_share * (r_beta - financing)
