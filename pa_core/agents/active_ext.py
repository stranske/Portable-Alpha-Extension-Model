from __future__ import annotations

from .types import Agent, Array


class ActiveExtensionAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        self._validate_inputs(r_beta, alpha_stream, financing)
        active_share = (
            float(self.extra.get("active_share", 50.0)) / 100.0
        )  # Convert percentage to decimal
        return (
            self.p.beta_share * (r_beta - financing)
            + (self.p.beta_share * active_share) * alpha_stream
        )
