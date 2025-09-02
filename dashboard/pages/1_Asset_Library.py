"""Asset Library page for uploading and calibrating asset data."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from pa_core.data import CalibrationAgent, DataImportAgent

try:
    # Detect bare (pytest) mode where Streamlit ScriptRunContext is absent
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

    _BARE_MODE = get_script_run_ctx() is None
except Exception:  # pragma: no cover - conservative fallback
    _BARE_MODE = True
from pa_core.presets import AlphaPreset, PresetLibrary
from dashboard.app import _DEF_THEME, apply_theme


def main() -> None:
    st.title("Asset Library")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return

    # Persist the uploaded file to a temp path for pandas/Excel readers
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        st.markdown("---")
        st.subheader("Column Mapping & Parse Options")

        # Mapping options (with simple defaults)
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.text_input("Date column", value="Date")
            wide = (
                st.radio(
                    "Input layout", options=["wide", "long"], index=0, horizontal=True
                )
                == "wide"
            )
            value_type = st.radio(
                "Value type", options=["returns", "prices"], index=0, horizontal=True
            )
        with col2:
            id_col = st.text_input("ID column (long only)", value="Id")
            value_col = st.text_input("Value column (long only)", value="Return")
            frequency = st.radio(
                "Frequency", options=["monthly", "daily"], index=0, horizontal=True
            )
        with col3:
            min_obs = st.number_input(
                "Min observations per id", min_value=1, value=36, step=1
            )
            sheet_name = st.text_input("Excel sheet name (optional)", value="")
            na_values_str = st.text_input("NA markers (comma‑sep)", value="")
            decimal = st.text_input("Decimal separator", value=".", max_chars=1)
            thousands = st.text_input("Thousands separator", value="", max_chars=1)

        # Build importer with selected options
        na_values = [s.strip() for s in na_values_str.split(",") if s.strip()] or None
        sheet_arg = sheet_name or None
        thousands_arg = thousands or None

        importer = DataImportAgent(
            date_col=date_col,
            id_col=id_col,
            value_col=value_col,
            wide=wide,
            value_type=value_type,  # "returns" | "prices"
            frequency=frequency,  # "monthly" | "daily"
            min_obs=int(min_obs),
            sheet_name=sheet_arg,
            na_values=na_values,
            decimal=decimal or ".",
            thousands=thousands_arg,
        )

        # Template save/load helpers
        if not _BARE_MODE:
            st.markdown("**Mapping Templates**")
            tcol1, tcol2 = st.columns([1, 1])
            with tcol1:
                if st.button("💾 Save mapping as template"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as t:
                        importer.save_template(t.name)
                        tmpl_str = Path(t.name).read_text()
                    st.download_button(
                        "Download mapping.yaml",
                        tmpl_str,
                        file_name="mapping.yaml",
                        mime="application/x-yaml",
                    )
            with tcol2:
                tmpl_upload = st.file_uploader(
                    "Load mapping template", type=["yaml", "yml"], key="import_mapping"
                )
                if tmpl_upload is not None and st.button("Apply template"):
                    from pa_core.data import DataImportAgent as _DIA

                    text = tmpl_upload.getvalue().decode("utf-8")
                    # Write to temp so from_template can read
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as t:
                        Path(t.name).write_text(text)
                        importer = _DIA.from_template(t.name)

        # Load using current importer configuration
        df = importer.load(tmp_path)
        st.success("Data loaded successfully with current mapping")
        st.dataframe(df, use_container_width=True)
        st.json(importer.metadata)

        # Calibration flow remains the same
        ids = sorted(df["id"].unique())
        index_id = st.selectbox("Index column", ids)
        if st.button("Calibrate"):
            calib = CalibrationAgent(min_obs=importer.min_obs)
            result = calib.calibrate(df, index_id)
            yfd, ypath = tempfile.mkstemp(suffix=".yaml")
            try:
                calib.to_yaml(result, ypath)
                yaml_str = Path(ypath).read_text()
                st.download_button(
                    "Download Asset Library YAML",
                    yaml_str,
                    file_name="asset_library.yaml",
                    mime="application/x-yaml",
                )
            finally:
                try:
                    os.close(yfd)
                except Exception:
                    pass
                Path(ypath).unlink(missing_ok=True)

        # Preset library management
        st.subheader("Alpha Presets")
        lib: PresetLibrary = st.session_state.setdefault("preset_lib", PresetLibrary())
        if lib.presets:
            st.dataframe(
                [asdict(p) for p in lib.presets.values()],
                use_container_width=True,
            )
        with st.form("preset_form"):
            pid = st.text_input("Preset ID")
            mu = st.number_input(
                "mu [annual %]",
                value=0.0,
                format="%.4f",
                help="Expected annual return as a decimal (e.g., 0.04 for 4%)",
            )
            sigma = st.number_input(
                "sigma [annual %]",
                value=0.0,
                format="%.4f",
                help="Annual volatility as a decimal (e.g., 0.10 for 10%)",
            )
            rho = st.number_input(
                "rho [-1..1]",
                value=0.0,
                format="%.4f",
                help="Correlation coefficient between -1 and 1",
            )
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
        st.download_button(
            "Download Presets JSON",
            lib.to_json_str(),
            file_name="alpha_presets.json",
            mime="application/json",
        )
        uploaded_preset = st.file_uploader(
            "Import presets", type=["yaml", "yml", "json"], key="import_presets"
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
