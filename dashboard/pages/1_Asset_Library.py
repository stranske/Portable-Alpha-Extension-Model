"""Asset Library page for uploading and calibrating asset data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from pa_core.data import DataImportAgent
from dashboard.app import _DEF_THEME, apply_theme


def main() -> None:
    st.title("Asset Library")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    importer = DataImportAgent()
    df = importer.load(tmp_path)
    st.dataframe(df)
    st.json(importer.metadata)


if __name__ == "__main__":
    main()
