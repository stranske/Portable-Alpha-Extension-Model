"""Guided interface for running a full simulation with validation."""

from __future__ import annotations

import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

import streamlit as st

from dashboard.app import _DEF_THEME, _DEF_XLSX, apply_theme
from dashboard.validation_ui import (
    display_validation_results,
    create_validation_sidebar, 
    validate_scenario_config,
    create_margin_buffer_display,
    validation_status_indicator
)
from pa_core import cli as pa_cli


def _write_temp(uploaded: st.runtime.uploaded_file_manager.UploadedFile, suffix: str) -> str:
    """Write *uploaded* content to a temporary file with *suffix* and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def main() -> None:
    st.title("Scenario Wizard")
    
    # Theme and validation settings
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    validation_settings = create_validation_sidebar()

    st.write("Upload a scenario configuration and index returns to run a simulation.")
    
    # File uploaders
    cfg = st.file_uploader("Scenario YAML", type=["yaml", "yml"])
    idx = st.file_uploader("Index CSV", type=["csv"])
    output = st.text_input("Output workbook", _DEF_XLSX)
    
    # Display validation results if config is uploaded
    if cfg is not None and validation_settings.get('validate_on_change', True):
        st.subheader("📋 Configuration Validation")
        
        try:
            # Parse uploaded YAML content
            config_data = yaml.safe_load(cfg.getvalue())
            
            # Validate the configuration
            validation_results = validate_scenario_config(config_data, validation_settings)
            
            # Display validation results
            display_validation_results(validation_results, "Configuration Validation")
            
            # Display margin buffer if capital allocation data is available
            if all(key in config_data for key in ['external_pa_capital', 'active_ext_capital', 'internal_pa_capital']):
                try:
                    from pa_core.validators import calculate_margin_requirement
                    
                    reference_sigma = config_data.get('reference_sigma', 0.01) 
                    volatility_multiple = config_data.get('volatility_multiple', 3.0)
                    total_capital = config_data.get('total_fund_capital', 1000.0)
                    
                    margin_requirement = calculate_margin_requirement(
                        reference_sigma=reference_sigma,
                        volatility_multiple=volatility_multiple, 
                        total_capital=total_capital
                    )
                    
                    internal_pa_capital = config_data.get('internal_pa_capital', 0.0)
                    available_buffer = total_capital - margin_requirement - internal_pa_capital
                    
                    create_margin_buffer_display(margin_requirement, available_buffer, total_capital)
                    
                except Exception as e:
                    st.warning(f"Could not compute margin requirements: {e}")
            
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML format: {e}")
        except Exception as e:
            st.warning(f"Could not validate configuration: {e}")
    
    # Run button with validation status
    validation_status = "✅" 
    run_disabled = False
    
    if cfg is not None and validation_settings.get('validate_on_change', True):
        try:
            config_data = yaml.safe_load(cfg.getvalue())
            validation_results = validate_scenario_config(config_data, validation_settings)
            validation_status = validation_status_indicator(validation_results)
            
            # Disable run if there are validation errors
            has_errors = any(not r.is_valid for r in validation_results)
            if has_errors:
                run_disabled = True
                st.warning("⚠️ Cannot run simulation due to validation errors. Please resolve the issues shown above.")
                
        except Exception:
            validation_status = "❓"

    if st.button(f"{validation_status} Run Simulation", disabled=run_disabled):
        if cfg is None or idx is None:
            st.warning("Please upload both files before running.")
            return

        cfg_path = _write_temp(cfg, ".yaml")
        idx_path = _write_temp(idx, ".csv")
        try:
            with st.spinner("Running simulation..."):
                pa_cli.main(["--config", cfg_path, "--index", idx_path, "--output", output])
        except Exception as exc:  # pragma: no cover - runtime feedback
            st.error(f"Simulation failed: {exc}")
        else:
            st.success(f"✅ Run complete! Results written to {output}.")
            st.page_link("pages/4_Results.py", label="View results →")
        finally:
            Path(cfg_path).unlink(missing_ok=True)
            Path(idx_path).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
