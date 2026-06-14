from __future__ import annotations

from ..config import normalize_share
from .types import Agent, Array


class ActiveExtensionAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        self._validate_inputs(r_beta, alpha_stream, financing)
        active_share = normalize_share(self.extra.get("active_share", 0.5))
        if active_share is None:
            active_share = 0.0
        s = float(active_share)
        # Transfer-coefficient diminishing returns (issue #1924). decay == 0 -> tc == 1
        # -> the haircut and cost terms below are exactly 0.0, so the legacy linear
        # formula is reproduced bit-for-bit. decay > 0 makes expected alpha concave in s
        # (declining information ratio) while active risk still scales linearly.
        decay = float(self.extra.get("tc_decay", 0.0))
        alpha_mu = float(self.extra.get("alpha_mu", 0.0))
        cost_per_share = float(self.extra.get("cost_per_share", 0.0))
        tc = 1.0 / (1.0 + decay * s)
        weight = self.p.beta_share * s
        return (
            self.p.beta_share * (r_beta - financing)
            + weight * alpha_stream
            - weight * (1.0 - tc) * alpha_mu
            - self.p.beta_share * (cost_per_share * s)
        )
