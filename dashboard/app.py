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


def _load_paths_sidecar(xlsx: str) -> pd.DataFrame | None:
    """Best-effort loader for path data beside an Excel file.

    Order of attempts:
    1) Read sibling .parquet if present (guarded for missing engines).
    2) Read sibling .csv if present.
    3) Read 'AllReturns' sheet from the Excel workbook.

    Returns None if no option succeeds.
    """
    # 1) Parquet sidecar
    p_parquet = Path(xlsx).with_suffix(".parquet")
    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            # Graceful degradation if pyarrow/fastparquet is missing or file invalid
            pass

    # 2) CSV sidecar
    p_csv = Path(xlsx).with_suffix(".csv")
    if p_csv.exists():
        try:
            return pd.read_csv(p_csv)
        except Exception:
            pass

    # 3) Excel 'AllReturns' sheet fallback
    try:
        return pd.read_excel(xlsx, sheet_name="AllReturns")
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_data(xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    summary = pd.read_excel(xlsx, sheet_name="Summary")
    p = Path(xlsx).with_suffix(".parquet")
    paths: pd.DataFrame | None = None
    if p.exists():
        try:
            paths = pd.read_parquet(p)
        except ImportError:
            csv_path = p.with_suffix(".csv")
            if csv_path.exists():
                st.info("pyarrow missing; using CSV path data instead")
                paths = pd.read_csv(csv_path, index_col=0)
            else:
                st.info(
                    "Install pyarrow for Parquet support or provide a matching CSV file"
                )
    return summary, paths


def save_history(df: pd.DataFrame, base: str | Path = "Outputs.parquet") -> None:
    """Persist run history to Parquet and CSV for dashboard use."""
    p = Path(base)
    try:
        df.to_parquet(p)
    except ImportError:
        st.info("pyarrow not installed; skipping Parquet export")
    df.to_csv(p.with_suffix(".csv"), index=False)


def load_history(parquet: str = "Outputs.parquet") -> pd.DataFrame | None:
    """Return mean and vol by simulation from ``parquet`` or CSV."""
    p = Path(parquet)
    df: pd.DataFrame | None = None
    if p.exists():
        try:
            df = pd.read_parquet(p)
        except ImportError:
            if csv_path.exists():
                st.info("pyarrow missing; loading CSV history")
                df = pd.read_csv(csv_path)
            else:
                st.info(
                    "Install pyarrow for Parquet support or provide a CSV fallback"
                )
                return None
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None
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
