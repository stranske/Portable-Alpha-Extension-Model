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
        self._validate_inputs(r_beta, alpha_stream, financing)
        return self.p.alpha_share * alpha_stream
