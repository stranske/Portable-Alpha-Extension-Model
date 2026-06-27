"""Shared utilities and home page for the Streamlit dashboard."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import streamlit as st

from dashboard.glossary import GLOSSARY
from dashboard.theme import apply_theme as apply_ds_theme
from dashboard.theme import ds
from dashboard.utils import bundled_sample_index_path
from pa_core.contracts import (
    ALL_RETURNS_SHEET_NAME,
    DEFAULT_OUTPUT_FILENAME,
    SUMMARY_SHEET_NAME,
)

PLOTS: dict[str, str] = {
    "Headline": "pa_core.viz.risk_return.make",
    "Funding fan": "pa_core.viz.fan.make",
    "Path dist": "pa_core.viz.path_dist.make",
}

_DEF_XLSX = DEFAULT_OUTPUT_FILENAME
_DEF_THEME = "config_theme.yaml"
_PARQUET_HINT = "Parquet support missing; install pyarrow or use CSV."

ADVANCED_SETTINGS_LABEL = "Advanced / Settings"


def render_settings_sidebar(results_default: str | None = None) -> tuple[str | None, str]:
    """Render file-path internals inside a collapsed sidebar expander.

    The theme-file path (and, on the Results page, the results workbook path)
    are configuration internals that previously sat as raw text inputs at the
    top of every page's sidebar. Tucking them inside a collapsed
    ``"Advanced / Settings"`` expander keeps the default sidebar focused on the
    user's workflow while leaving the controls available for power users.

    Returns ``(results_path, theme_path)``. When ``results_default`` is ``None``
    the results input is omitted and the returned results path is ``None``.
    """
    with st.sidebar.expander(ADVANCED_SETTINGS_LABEL, expanded=False):
        results_path = (
            st.text_input("Results file", results_default) if results_default is not None else None
        )
        theme_path = st.text_input("Theme file", _DEF_THEME)
    return results_path, theme_path


def _cache_data(ttl: int):
    """Fallback wrapper when Streamlit cache decorators are unavailable."""
    cache_fn = getattr(st, "cache_data", None)
    if cache_fn is None:

        def _decorator(func):
            return func

        return _decorator
    return cache_fn(ttl=ttl)


def _show_parquet_hint() -> None:
    st.info(_PARQUET_HINT)


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

    def _read_parquet_sidecar(path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_parquet(path)
        except ImportError:
            _show_parquet_hint()
            return None
        except Exception:
            return None

    # 1) Parquet sidecar
    p_parquet = Path(xlsx).with_suffix(".parquet")
    if p_parquet.exists():
        parquet_df = _read_parquet_sidecar(p_parquet)
        if parquet_df is not None:
            return parquet_df

    # 2) CSV sidecar
    p_csv = Path(xlsx).with_suffix(".csv")
    if p_csv.exists():
        try:
            return pd.read_csv(p_csv)
        except Exception:
            pass

    # 3) Excel 'AllReturns' sheet fallback
    try:
        return pd.read_excel(xlsx, sheet_name=ALL_RETURNS_SHEET_NAME)
    except Exception:
        return None


@_cache_data(ttl=600)
def load_data(xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    summary = pd.read_excel(xlsx, sheet_name=SUMMARY_SHEET_NAME)
    # Try parquet -> csv -> Excel('AllReturns') for path-based charts
    paths = _load_paths_sidecar(xlsx)
    return summary, paths


def save_history(df: pd.DataFrame, base: str | Path = "Outputs.parquet") -> None:
    """Persist run history to Parquet and CSV for dashboard use."""
    p = Path(base)
    try:
        df.to_parquet(p)
    except ImportError:
        _show_parquet_hint()
    df.to_csv(p.with_suffix(".csv"), index=False)


# Provide test convenience: when running under pytest, expose save_history as a builtin
# so tests can call `save_history(...)` without importing it explicitly.
try:
    import builtins as _builtins
    import sys as _sys

    if "pytest" in _sys.modules:
        setattr(_builtins, "save_history", save_history)
except Exception:
    # Non-fatal; ignore if environment doesn't allow this registration
    pass


def load_history(parquet: str = "Outputs.parquet") -> pd.DataFrame | None:
    """Return mean and vol by simulation from ``parquet`` or CSV."""
    p = Path(parquet)
    df: pd.DataFrame | None = None
    if p.exists():
        try:
            df = pd.read_parquet(p)
        except ImportError:
            _show_parquet_hint()
            df = None
        except Exception:
            df = None
    # CSV fallback: try sibling .csv if parquet missing or unreadable
    if df is None:
        p_csv = p.with_suffix(".csv")
        if p_csv.exists():
            try:
                df = pd.read_csv(p_csv)
            except Exception:
                df = None
    if df is None:
        return None
    return (
        df.groupby("Sim")["Return"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_return", "std": "volatility"})
    )


def _render_getting_started() -> None:
    """Surface the bundled sample dataset so first-run users aren't upload-gated.

    The data-driven pages (Stress Lab, Scenario Grid) now offer a "Use bundled
    sample data" option, so a non-technical first-run user can run an end-to-end
    example in one click. This section advertises that and offers the bundled CSV
    as a download for users who prefer the explicit upload flow. See issue #1900.
    """
    sample_path = bundled_sample_index_path()
    if not sample_path.exists():
        return
    with st.expander("New here? Try the bundled sample data", expanded=True):
        st.markdown(
            "No data yet? The **Stress Lab** and **Scenario Grid** pages each have a "
            "**“Use bundled sample data (no upload needed)”** checkbox that loads the "
            "bundled S&P 500 TR / FRED dividend-yield series so you can run a complete "
            "example in one click — no upload required."
        )
        try:
            sample_bytes = sample_path.read_bytes()
        except OSError:
            return
        st.download_button(
            "Download sample index CSV",
            data=sample_bytes,
            file_name=sample_path.name,
            mime="text/csv",
            help="Use this with any page's upload control if you prefer the explicit flow.",
        )


def main() -> None:
    """Render the dashboard home page."""

    # Give the home page a real browser/page title instead of the Streamlit
    # default derived from the entry-point filename ("app").
    try:
        st.set_page_config(page_title="Portable Alpha Dashboard", page_icon="📈")
    except Exception:
        # set_page_config raises if called more than once per session (e.g. on
        # a Streamlit rerun) or when unavailable in bare/test mode; the title is
        # cosmetic, so a repeat call must not break the page.
        pass

    apply_ds_theme()

    st.title("Portable Alpha Dashboard")
    st.write("Select a page from the sidebar or use the links below to begin.")

    with st.sidebar.expander("Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** – {definition}")

    # Primary "start here" cue: the fastest path to a complete result is a
    # one-click bundled-sample run on Stress Lab (#2041 home discoverability).
    _ds = ds()
    if _ds is not None:
        _ds.notice(
            "info",
            "New here? Start with Stress Lab",
            "Tick “Use bundled sample data” on Stress Lab (or Scenario Grid) to run a "
            "complete example in one click — no upload needed.",
        )
    st.page_link("pages/6_Stress_Lab.py", label="Start here → Stress Lab (one-click example)")

    # Group the tools so first-timers can tell run paths from inputs/outputs, and
    # where each run's results appear.
    st.markdown("**Run a scenario**")
    st.caption("Stress Lab & Scenario Grid show results in-page; the Scenario Wizard writes to the Results page.")
    st.page_link("pages/6_Stress_Lab.py", label="Stress Lab (presets)")
    st.page_link("pages/5_Scenario_Grid.py", label="Scenario Grid & Frontier (beta)")
    st.page_link("pages/3_Scenario_Wizard.py", label="Scenario Wizard")

    st.markdown("**Inputs & outputs**")
    st.page_link("pages/1_Asset_Library.py", label="Asset Library")
    st.page_link("pages/2_Portfolio_Builder.py", label="Portfolio Builder")
    st.page_link("pages/4_Results.py", label="Results (Scenario Wizard output)")

    _render_getting_started()

    history = load_history()
    if history is not None:
        st.subheader("Run history")
        st.dataframe(history)
    else:
        st.info("No runs found. Run a scenario to see history.")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
