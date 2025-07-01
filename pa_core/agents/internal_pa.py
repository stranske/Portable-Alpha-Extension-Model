from __future__ import annotations
from .types import Agent, Array

class InternalPAAgent(Agent):
    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        """Return pure in-house alpha with input validation."""
        if (
            r_beta.shape != alpha_stream.shape
            or r_beta.shape != financing.shape
        ):
            raise ValueError("shape mismatch")
        return self.p.alpha_share * alpha_stream
