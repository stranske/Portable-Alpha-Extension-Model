"""Shared utilities and home page for the Streamlit dashboard."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import streamlit as st

from dashboard.glossary import GLOSSARY

PLOTS: dict[str, str] = {
    "Headline": "pa_core.viz.risk_return.make",
    "Funding fan": "pa_core.viz.fan.make",
    "Path dist": "pa_core.viz.path_dist.make",
}

_DEF_XLSX = "Outputs.xlsx"
_DEF_THEME = "config_theme.yaml"


def _load_theme() -> ModuleType:
    """Return ``pa_core.viz.theme`` loaded without package side effects."""
    theme_path = Path(__file__).resolve().parents[1] / "pa_core" / "viz" / "theme.py"
    spec = importlib.util.spec_from_file_location("pa_core.viz.theme", theme_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module 'pa_core.viz.theme' from {theme_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return importlib.import_module("pa_core.viz.theme")


def apply_theme(path: str) -> ModuleType:
    """Reload the dashboard colour palette and return the theme module."""
    theme = _load_theme()
    p = Path(path)
    if p.exists():
        theme.reload_theme(str(p))
    return theme


def _get_plot_fn(path: str):
    module, func = path.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, func)


@st.cache_data(ttl=600)
def load_data(xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    summary = pd.read_excel(xlsx, sheet_name="Summary")
    p = Path(xlsx).with_suffix(".parquet")
    paths = pd.read_parquet(p) if p.exists() else None
    return summary, paths


def load_history(parquet: str = "Outputs.parquet") -> pd.DataFrame | None:
    """Return mean and vol by simulation from ``parquet`` if it exists."""
    p = Path(parquet)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return (
        df.groupby("Sim")["Return"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_return", "std": "volatility"})
    )


def main() -> None:
    """Render the dashboard home page."""

    st.title("Portable Alpha Dashboard")
    st.write("Select a page from the sidebar or use the links below to begin.")

    with st.sidebar.expander("Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** – {definition}")

    st.page_link("pages/1_Asset_Library.py", label="Asset Library")
    st.page_link("pages/2_Portfolio_Builder.py", label="Portfolio Builder")
    st.page_link("pages/3_Scenario_Wizard.py", label="Scenario Wizard")
    st.page_link("pages/4_Results.py", label="Results")
    st.page_link("pages/5_Scenario_Grid.py", label="Scenario Grid & Frontier (beta)")
    st.page_link("pages/6_Stress_Lab.py", label="Stress Lab (presets)")

    history = load_history()
    if history is not None:
        st.subheader("Run history")
        st.dataframe(history)
    else:
        st.info("No runs found. Run a scenario to see history.")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
