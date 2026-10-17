"""Real Streamlit cleanup must not discard a provider's explicit override."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("panel", ["explain", "compare"])
def test_provider_roundtrip_preserves_override_after_real_widget_cleanup(
    panel, socket_connect_guard
):
    code = """
import pandas as pd
from dashboard.components import llm_settings
from dashboard.components.explain_results import render_explain_results_panel
from dashboard.components.comparison_llm import render_comparison_llm_panel
original_read = llm_settings.read_secret
llm_settings.read_secret = lambda name: "openai" if name == "PA_LLM_PROVIDER" else None
try:
    if PANEL == "explain":
        render_explain_results_panel(summary_df=pd.DataFrame({"monthly_TE": [0.02]}), manifest={}, xlsx_path="/tmp/results.xlsx")
    else:
        render_comparison_llm_panel(summary_df=pd.DataFrame({"monthly_TE": [0.02]}), manifest_data={}, run_key="roundtrip")
finally:
    llm_settings.read_secret = original_read
"""
    app = AppTest.from_string(code.replace("PANEL", repr(panel))).run()
    assert not app.exception
    api_key = next(item for item in app.text_input if item.label == "API Key")
    api_key.input("synthetic-typed-openai").run()
    provider = next(item for item in app.selectbox if item.label == "Provider")
    provider.select("anthropic").run()
    assert not app.exception
    assert next(item for item in app.text_input if item.label == "API Key").value == ""
    next(item for item in app.text_input if item.label == "API Key").input(
        "synthetic-typed-anthropic"
    ).run()
    next(item for item in app.selectbox if item.label == "Provider").select("openai").run()
    assert not app.exception
    assert (
        next(item for item in app.text_input if item.label == "API Key").value
        == "synthetic-typed-openai"
    )
    next(item for item in app.selectbox if item.label == "Provider").select("anthropic").run()
    assert not app.exception
    assert (
        next(item for item in app.text_input if item.label == "API Key").value
        == "synthetic-typed-anthropic"
    )
    assert socket_connect_guard[0] == []
