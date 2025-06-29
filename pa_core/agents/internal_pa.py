from __future__ import annotations
from .types import Agent, Array

class InternalPAAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        if r_beta.shape != alpha_stream.shape:
            raise ValueError("shape mismatch")
        return self.p.alpha_share * alpha_stream
