"""Documentation contracts for optional dashboard LLM features."""

from __future__ import annotations

from pathlib import Path


def test_llm_features_doc_names_install_env_and_reference_pack() -> None:
    text = Path("docs/llm_features.md").read_text()

    assert 'python -m pip install -e ".[llm]"' in text
    for env_var in (
        "PA_LLM_PROVIDER",
        "PA_LLM_MODEL",
        "PA_STREAMLIT_API_KEY",
        "OPENAI_API_KEY",
        "CLAUDE_API_STRANSKE",
        "PA_LLM_BASE_URL",
        "PA_LLM_API_VERSION",
        "AZURE_OPENAI_API_VERSION",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ):
        assert env_var in text
    assert ".reference/trend_streamlit_llm/" in text
    assert "must not be committed" in text


def test_readme_links_llm_features_doc() -> None:
    readme = Path("README.md").read_text()

    assert "[Streamlit LLM Features](docs/llm_features.md)" in readme
