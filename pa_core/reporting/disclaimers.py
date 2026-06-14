"""Shared model-limitation disclaimers.

Single source of truth for the caveats that must accompany any committee-facing
output (the README, the PPTX board pack, and exported packets). Keeping the text
in one place ensures the documentation and the generated board pack cannot drift
apart. See issue #1923.
"""

from __future__ import annotations

LIMITATIONS_TITLE = "Model Limitations & Caveats"

#: Ordered list of model caveats. Each entry is a single, self-contained
#: statement suitable for a README bullet or a slide line.
MODEL_LIMITATIONS: tuple[str, ...] = (
    "Results are gross of fees and costs.",
    "Total excludes Base (overlay semantics): a Base-only fund shows Total = 0.",
    "Monthly draws are i.i.d. — no volatility clustering is modelled.",
    "Regimes are ignored in parameter sweeps.",
    "Financing `broadcast` reuses a single financing path across simulations.",
    "The model is forward-looking and has not been backtested.",
    "`risk_metrics` is advisory: it selects which metrics are reported, "
    "not how the simulation runs.",
    "`Scenario.sleeves` is currently unwired and does not affect simulation "
    "results.",
)


def limitations_markdown() -> str:
    """Return the limitations as a Markdown bullet list."""
    return "\n".join(f"- {item}" for item in MODEL_LIMITATIONS)
