from __future__ import annotations
from .types import Agent, Array

class ActiveExtensionAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape or r_beta.shape != financing.shape:
            raise ValueError("shape mismatch")
        active_share = float(self.extra.get("active_share", 0.5))
        return (
            self.p.beta_share * (r_beta - financing)
            + (self.p.beta_share * active_share) * alpha_stream
        )
