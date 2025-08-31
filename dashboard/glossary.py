"""Glossary of finance terms for dashboard tooltips."""

GLOSSARY = {
    "active share": "Portion of a portfolio that differs from its benchmark; 0% means identical holdings, 100% fully independent.",
    "buffer multiple": "Multiplier applied to volatility to set the cash buffer threshold for drawdowns.",
    "breach probability": "Likelihood that returns fall below the buffer threshold in a given period.",
    "TE": "Tracking error — standard deviation of portfolio returns minus benchmark returns.",
    "CVaR": "Conditional Value at Risk — expected loss given that losses exceed the VaR cutoff.",
}


def tooltip(term: str) -> str:
    """Return plain-English definition for ``term`` if available."""
    return GLOSSARY.get(term.lower(), GLOSSARY.get(term, ""))
