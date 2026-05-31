from __future__ import annotations

from .types import Agent, Array


class InternalPAAgent(Agent):
    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        """Return in-house alpha net of internal-PA financing cost.

        ``financing`` is the internal-PA financing cost series (issue #1849):
        a positive value is a cost that lowers the sleeve return, a negative
        value is a benefit/positive carry that raises it. It is subtracted at
        the sleeve level; total-portfolio scaling by the sleeve's capital share
        is handled by the contribution machinery. When internal-PA financing is
        disabled (the default), ``financing`` is all-zeros and the return is
        pure in-house alpha, preserving historical behaviour.
        """
        self._validate_inputs(r_beta, alpha_stream, financing)
        return self.p.alpha_share * alpha_stream - financing
