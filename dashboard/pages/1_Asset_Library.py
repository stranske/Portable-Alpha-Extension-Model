"""Asset Library page for uploading and calibrating asset data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from pa_core.data import CalibrationAgent, DataImportAgent
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
    try:
        importer = DataImportAgent()
        df = importer.load(tmp_path)
        st.dataframe(df)
        st.json(importer.metadata)

        ids = sorted(df["id"].unique())
        index_id = st.selectbox("Index column", ids)
        if st.button("Calibrate"):
            calib = CalibrationAgent(min_obs=importer.min_obs)
            result = calib.calibrate(df, index_id)
            tmp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            try:
                calib.to_yaml(result, tmp_yaml.name)
                yaml_str = Path(tmp_yaml.name).read_text()
                st.download_button(
                    "Download Asset Library YAML",
                    yaml_str,
                    file_name="asset_library.yaml",
                    mime="application/x-yaml",
                )
            finally:
                # Clean up the YAML temp file
                Path(tmp_yaml.name).unlink(missing_ok=True)
    finally:
        # Clean up the uploaded data temp file
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
