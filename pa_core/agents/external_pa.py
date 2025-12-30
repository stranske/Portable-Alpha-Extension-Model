from __future__ import annotations

from .types import Agent, Array


class ExternalPAAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        self._validate_inputs(r_beta, alpha_stream, financing)
        theta = float(self.extra.get("theta_extpa", 0.0))
        return self.p.beta_share * (r_beta - financing) + (self.p.beta_share * theta) * alpha_stream
