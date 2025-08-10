"""Placeholder page for guided scenario setup."""

from __future__ import annotations

import streamlit as st

from dashboard.app import _DEF_THEME, apply_theme


def main() -> None:
    st.title("Scenario Wizard")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    st.write("Coming soon: guided scenario configuration.")


if __name__ == "__main__":
    main()
