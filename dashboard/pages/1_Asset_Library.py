"""Asset Library page for uploading and calibrating asset data."""

from __future__ import annotations

import logging
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
from dashboard.app import _DEF_THEME, apply_theme
from pa_core.presets import AlphaPreset, PresetLibrary

# Create logger for this module
logger = logging.getLogger(__name__)


def main() -> None:
    st.title("Asset Library")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    st.markdown("**Upload asset return data**")
    st.caption("Drag and drop a CSV or Excel file, or click to browse.")
    uploaded = st.file_uploader(
        "Drag-and-drop CSV/XLSX",
        type=["csv", "xlsx", "xls"],
        help="CSV and Excel files are supported.",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return
    data = uploaded.getvalue()
    size_kb = len(data) / 1024
    st.caption(f"Selected file: {uploaded.name} ({size_kb:.1f} KB)")

    # Persist the uploaded file to a temp path for pandas/Excel readers
    suffix = Path(uploaded.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
            tmp.flush()  # Ensure data is written to disk

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
            monthly_rule = "ME"
            if frequency == "daily":
                monthly_rule = st.radio(
                    "Monthly resample",
                    options=["ME", "MS"],
                    index=0,
                    horizontal=True,
                    format_func=lambda rule: (
                        "Month-end [ME]" if rule == "ME" else "Month-start [MS]"
                    ),
                    help="Choose the month label for daily-to-monthly compounding.",
                )
        with col3:
            min_obs = st.number_input(
                "Min observations per id", min_value=1, value=36, step=1
            )
            cov_shrinkage = st.selectbox(
                "Covariance shrinkage",
                options=["none", "ledoit_wolf"],
                index=0,
                help="Ledoit-Wolf shrinkage improves stability in short samples.",
            )
            vol_regime = st.selectbox(
                "Volatility regime",
                options=["single", "two_state"],
                index=0,
                help="Two-state uses recent window to select high/low vol.",
            )
            regime_window = st.number_input(
                "Vol regime window (months)",
                min_value=2,
                value=12,
                step=1,
                help="Recent window length for two-state regime selection.",
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
            monthly_rule=monthly_rule,  # "ME" | "MS"
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
                    # Persist current importer mapping to a YAML template (secure temp file)
                    fd, tmp_tmpl = tempfile.mkstemp(suffix=".yaml")
                    try:
                        importer.save_template(tmp_tmpl)
                        tmpl_str = Path(tmp_tmpl).read_text()
                    finally:
                        try:
                            os.close(fd)
                        except OSError as e:
                            # Log the error but don't fail - file descriptor might already be closed
                            logger.warning(f"Error closing file descriptor {fd}: {e}")
                        Path(tmp_tmpl).unlink(missing_ok=True)
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
                    # Write to temp so from_template can read (secure temp file)
                    fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml")
                    try:
                        with os.fdopen(fd, "wb") as t:
                            t.write(text.encode("utf-8"))
                            t.flush()
                        importer = _DIA.from_template(tmp_yaml)
                    finally:
                        Path(tmp_yaml).unlink(missing_ok=True)

        # Load using current importer configuration
        df = importer.load(tmp_path)
        st.success("Data loaded successfully with current mapping")
        st.dataframe(df, use_container_width=True)
        st.json(importer.metadata)

        # Calibration flow remains the same
        ids = sorted(df["id"].unique())
        index_id = st.selectbox("Index column", ids)
        if st.button("Calibrate"):
            calib = CalibrationAgent(
                min_obs=importer.min_obs,
                covariance_shrinkage=cov_shrinkage,
                vol_regime=vol_regime,
                vol_regime_window=int(regime_window),
            )
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
                if result.diagnostics is not None:
                    st.json(
                        {
                            "covariance_shrinkage": result.diagnostics.covariance_shrinkage,
                            "shrinkage_intensity": result.diagnostics.shrinkage_intensity,
                            "vol_regime": result.diagnostics.vol_regime,
                            "vol_regime_window": result.diagnostics.vol_regime_window,
                            "vol_regime_state": result.diagnostics.vol_regime_state,
                        }
                    )
            finally:
                try:
                    os.close(yfd)
                except OSError as e:
                    # Log the error but don't fail - file descriptor might already be closed
                    logger.warning(f"Error closing file descriptor {yfd}: {e}")
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
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
