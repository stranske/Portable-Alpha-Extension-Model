"""Home page for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.app import _DEF_THEME, apply_theme


def main() -> None:
    """Render the landing page and theme selector."""
    st.title("Portable Alpha Dashboard")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    st.write("Use the sidebar to navigate between pages.")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
