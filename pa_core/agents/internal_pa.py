from __future__ import annotations

from .types import Agent, Array


class InternalPAAgent(Agent):
    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        """Return in-house alpha net of internal PA financing cost."""
        self._validate_inputs(r_beta, alpha_stream, financing)
        return self.p.alpha_share * (alpha_stream - financing)
