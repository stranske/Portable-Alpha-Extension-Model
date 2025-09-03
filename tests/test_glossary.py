from dashboard.glossary import GLOSSARY, tooltip

TERMS = ["active share", "buffer multiple", "breach probability", "TE", "CVaR"]


def test_glossary_terms_present() -> None:
    for term in TERMS:
        assert term in GLOSSARY
        assert GLOSSARY[term]


def test_tooltip_lookup_case_insensitive() -> None:
    assert tooltip("Active Share") == GLOSSARY["active share"]
