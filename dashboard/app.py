import importlib
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from pa_core.viz import theme

PLOTS: dict[str, str] = {
    "Headline": "pa_core.viz.risk_return.make",
    "Funding fan": "pa_core.viz.fan.make",
    "Path dist": "pa_core.viz.path_dist.make",
}

_DEF_XLSX = "Outputs.xlsx"
_DEF_THEME = "config_theme.yaml"


def apply_theme(path: str) -> None:
    """Reload the dashboard colour palette."""
    p = Path(path)
    if p.exists():
        theme.reload_theme(str(p))


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


def main() -> None:
    st.title("Portable Alpha Dashboard")
    xlsx = st.sidebar.text_input("Results file", _DEF_XLSX)
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    if not Path(xlsx).exists():
        st.warning(f"File {xlsx} not found")
        st.stop()
    summary, paths = load_data(xlsx)

    months = st.sidebar.slider("Months", 1, summary.shape[0], summary.shape[0])
    agents = st.sidebar.multiselect(
        "Agents", summary["Config"].unique().tolist(), summary["Config"].unique().tolist()
    )
    st.sidebar.number_input("Risk-free rate", value=0.0)
    auto = st.sidebar.checkbox("Auto-refresh")
    interval = st.sidebar.number_input("Refresh every (s)", 5, 300, 60)

    tab_names = list(PLOTS) + ["Diagnostics"]
    tabs = st.tabs(tab_names)

    for name, tab in zip(tab_names[:-1], tabs[:-1]):
        fn = _get_plot_fn(PLOTS[name])
        with tab:
            if name == "Headline":
                st.plotly_chart(fn(summary), use_container_width=True)
            elif paths is not None:
                sub_paths = paths[agents].iloc[:, :months]
                st.plotly_chart(fn(sub_paths), use_container_width=True)

    with tabs[-1]:
        st.dataframe(summary)

    png = _get_plot_fn(PLOTS["Headline"])(summary).to_image(format="png")
    st.download_button("Download PNG", png, file_name="risk_return.png", mime="image/png")
    with open(xlsx, "rb") as fh:
        st.download_button("Download XLSX", fh, file_name=Path(xlsx).name)

    if auto:
        time.sleep(interval)
        st.experimental_rerun()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
