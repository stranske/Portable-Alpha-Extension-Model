"""Asset Library page for uploading and calibrating asset data."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from pa_core.data import CalibrationAgent, DataImportAgent
from pa_core.presets import AlphaPreset, PresetLibrary
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

            st.subheader("Alpha Presets")
            lib: PresetLibrary = st.session_state.setdefault(
                "preset_lib", PresetLibrary()
            )
            if lib.presets:
                st.dataframe(
                    [asdict(p) for p in lib.presets.values()],
                    use_container_width=True,
                )
            with st.form("preset_form"):
                pid = st.text_input("Preset ID")
                mu = st.number_input("mu [annual %]", value=0.0, format="%.4f", help="Expected annual return as a decimal (e.g., 0.04 for 4%)")
                sigma = st.number_input("sigma [annual %]", value=0.0, format="%.4f", help="Annual volatility as a decimal (e.g., 0.10 for 10%)")
                rho = st.number_input("rho [-1..1]", value=0.0, format="%.4f", help="Correlation coefficient between -1 and 1")
                if st.form_submit_button("Save Preset") and pid:
                    preset = AlphaPreset(id=pid, mu=mu, sigma=sigma, rho=rho)
                    if pid in lib.presets:
                        lib.update(preset)
                    else:
                        lib.add(preset)
            del_id = st.selectbox("Delete preset", [""] + list(lib.presets), index=0)
            if st.button("Delete Preset") and del_id:
                lib.delete(del_id)
            st.download_button(
                "Download Presets YAML",
                lib.to_yaml_str(),
                file_name="alpha_presets.yaml",
                mime="application/x-yaml",
            )
            uploaded_preset = st.file_uploader(
                "Import presets", type=["yaml", "yml", "json"]
            )
            if uploaded_preset is not None:
                text = uploaded_preset.getvalue().decode("utf-8")
                if uploaded_preset.name.endswith((".yaml", ".yml")):
                    lib.load_yaml_str(text)
                else:
                    lib.load_json_str(text)
    finally:
        # Clean up the uploaded data temp file
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
