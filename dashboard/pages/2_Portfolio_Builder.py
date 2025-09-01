"""Interactive page for building portfolios from an asset library."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import yaml

from dashboard.app import _DEF_THEME, apply_theme
from pa_core.schema import Portfolio, load_scenario
from pa_core.portfolio.aggregator import PortfolioAggregator


def main() -> None:
    st.title("Portfolio Builder")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    uploaded = st.file_uploader("Upload Asset Library YAML", type=["yaml", "yml"])
    if uploaded is None:
        st.info("Upload an asset library YAML to begin.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        scenario = load_scenario(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not scenario.assets:
        st.warning("No assets found in file.")
        return

    st.subheader("Weights [sum to 1.0]") 
    weight_inputs: dict[str, float] = {}
    for asset in scenario.assets:
        weight_inputs[asset.id] = st.number_input(
            f"{asset.id} [0..1]", min_value=0.0, max_value=1.0, step=0.01, value=0.0,
        )
    total = sum(weight_inputs.values())
    if total == 0:
        st.warning("Enter weights for at least one asset.")
        return
    weights = {k: v / total for k, v in weight_inputs.items() if v > 0}
    st.write(f"Weights normalised to sum to 1 (total input={total:.2f}).")

    if st.button("Generate Portfolio"):
        try:
            scenario.portfolios = [Portfolio(id="portfolio1", weights=weights)]
            yaml_str = yaml.safe_dump(scenario.model_dump())
            st.download_button(
                "Download Portfolio YAML",
                yaml_str,
                file_name="portfolio.yaml",
                mime="application/x-yaml",
            )

            agg = PortfolioAggregator(scenario.assets, scenario.correlations)
            mu, sigma = agg.aggregate(weights)
            st.write(f"Expected return: {mu:.2%}")
            st.write(f"Expected volatility: {sigma:.2%}")
        except Exception as exc:  # pragma: no cover - streamlit runtime
            st.error(str(exc))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
