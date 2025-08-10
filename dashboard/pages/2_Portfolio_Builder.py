"""Placeholder page for portfolio construction."""

from __future__ import annotations

import streamlit as st

from dashboard.app import _DEF_THEME, apply_theme


def main() -> None:
    st.title("Portfolio Builder")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    st.write("Coming soon: interactive portfolio builder.")


if __name__ == "__main__":
    main()
