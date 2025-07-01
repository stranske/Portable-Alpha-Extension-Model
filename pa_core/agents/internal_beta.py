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
        self._validate_inputs(r_beta, alpha_stream, financing)
        return self.p.beta_share * (r_beta - financing)
