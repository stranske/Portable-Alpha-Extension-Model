"""Interactive page for building portfolios from an asset library."""

from __future__ import annotations

import tempfile

import streamlit as st
import yaml

from dashboard.app import _DEF_THEME, apply_theme
from dashboard.utils import normalize_share
from pa_core.portfolio.aggregator import PortfolioAggregator
from pa_core.schema import Portfolio, load_scenario


def main() -> None:
    st.title("Portfolio Builder")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    # Show any promoted alpha shares from Scenario Grid
    promoted_active_share: float | None = None
    promoted_theta: float | None = None
    promoted_source: str | None = None
    vals = None
    if "scenario_grid_selection" in st.session_state:
        vals = st.session_state["scenario_grid_selection"]
        promoted_source = "Scenario Grid selection"
    elif "promoted_alpha_shares" in st.session_state:
        vals = st.session_state["promoted_alpha_shares"]
        promoted_source = "Scenario Grid promotion"
    if vals is not None:
        try:
            promoted_active_share = normalize_share(float(vals["active_share"]))
            promoted_theta = normalize_share(float(vals["theta_extpa"]))
        except (TypeError, ValueError, KeyError):
            promoted_active_share = None
            promoted_theta = None
        if (promoted_active_share is not None) or (promoted_theta is not None):
            label = promoted_source or "Scenario Grid"
            st.info(
                (
                    f"{label} "
                    f"(active_share={(promoted_active_share or 0.0):.2f}, "
                    f"theta_extpa={(promoted_theta or 0.0):.2f})"
                )
            )

    # Apply promoted values to widget state once per promotion.
    promoted_state = (promoted_source, promoted_active_share, promoted_theta)
    last_promoted = st.session_state.get("alpha_shares_last_promoted")
    promotion_token = st.session_state.get("scenario_grid_promotion_token")
    last_token = st.session_state.get("alpha_shares_last_promotion_token")
    if promoted_active_share is not None and promoted_theta is not None and (
        promoted_state != last_promoted or promotion_token != last_token
    ):
        st.session_state["alpha_shares_active_share"] = promoted_active_share
        st.session_state["alpha_shares_theta_extpa"] = promoted_theta
        st.session_state["alpha_shares_last_promoted"] = promoted_state
        if promotion_token is not None:
            st.session_state["alpha_shares_last_promotion_token"] = promotion_token
            if promotion_token != last_token:
                st.rerun()

    # Optional alpha-share annotation (pre-populated when promoted)
    with st.expander(
        "Alpha Shares (annotation – included in download)",
        expanded=bool(promoted_active_share or promoted_theta),
    ):
        active_share_input = st.number_input(
            "Active share [0..1]",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=(promoted_active_share if promoted_active_share is not None else 0.5),
            key="alpha_shares_active_share",
            help="Optional. Included in the exported YAML under alpha_shares metadata.",
        )
        theta_extpa_input = st.number_input(
            "External PA alpha fraction θ [0..1]",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=(promoted_theta if promoted_theta is not None else 0.5),
            key="alpha_shares_theta_extpa",
            help="Optional. Included in the exported YAML under alpha_shares metadata.",
        )

    uploaded = st.file_uploader("Upload Asset Library YAML", type=["yaml", "yml"])
    if uploaded is None:
        st.info("Upload an asset library YAML to begin.")
        return

    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()  # Ensure data is written to disk
        scenario = load_scenario(tmp.name)

    if not scenario.assets:
        st.warning("No assets found in file.")
        return

    st.subheader("Weights [sum to 1.0]")
    weight_inputs: dict[str, float] = {}
    for asset in scenario.assets:
        weight_inputs[asset.id] = st.number_input(
            f"{asset.id} [0..1]",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.0,
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
            data = scenario.model_dump()
            # Attach alpha-share annotations if provided
            data["alpha_shares"] = {
                "active_share": float(active_share_input),
                "theta_extpa": float(theta_extpa_input),
            }
            yaml_str = yaml.safe_dump(data, sort_keys=False)
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
