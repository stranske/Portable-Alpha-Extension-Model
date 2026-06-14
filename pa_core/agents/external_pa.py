from __future__ import annotations

from ..config import normalize_share
from .types import Agent, Array


class ExternalPAAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        self._validate_inputs(r_beta, alpha_stream, financing)
        theta = normalize_share(self.extra.get("theta_extpa", 0.0))
        if theta is None:
            theta = 0.0
        t = float(theta)
        # Transfer-coefficient diminishing returns (issue #1924); decay == 0 reproduces
        # the legacy linear formula bit-for-bit (haircut and cost terms are exactly 0.0).
        decay = float(self.extra.get("tc_decay", 0.0))
        alpha_mu = float(self.extra.get("alpha_mu", 0.0))
        cost_per_share = float(self.extra.get("cost_per_share", 0.0))
        tc = 1.0 / (1.0 + decay * t)
        weight = self.p.beta_share * t
        return (
            self.p.beta_share * (r_beta - financing)
            + weight * alpha_stream
            - weight * (1.0 - tc) * alpha_mu
            - self.p.beta_share * (cost_per_share * t)
        )
