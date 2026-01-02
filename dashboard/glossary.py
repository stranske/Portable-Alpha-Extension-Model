"""Glossary of finance terms for dashboard tooltips."""

GLOSSARY = {
    "active share": "Portion of a portfolio that differs from its benchmark; 0% means identical holdings, 100% fully independent.",
    "buffer multiple": "Multiplier applied to volatility to set the cash buffer threshold for drawdowns.",
    "breach probability": "Share of simulated months across all paths that fall below the breach threshold.",
    "monthly_TE": "Tracking error — annualised volatility of active returns (portfolio minus benchmark).",
    "monthly_CVaR": "Conditional Value at Risk — expected loss given that losses exceed the VaR cutoff.",
    "monthly_MaxDD": "Worst peak-to-trough decline of the compounded wealth path.",
    "monthly_TimeUnderWater": "Fraction of periods where the compounded return is below zero.",
    "terminal_ShortfallProb": "Probability that the terminal compounded return is below the annualised threshold.",
}


def tooltip(term: str) -> str:
    """Return plain-English definition for ``term`` if available."""
    return GLOSSARY.get(term.lower(), GLOSSARY.get(term, ""))
