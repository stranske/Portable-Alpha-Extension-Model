from __future__ import annotations

from .types import Agent, Array


class BaseAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        self._validate_inputs(r_beta, alpha_stream, financing)
        return self.p.beta_share * (r_beta - financing) + self.p.alpha_share * alpha_stream
