"""Validation utilities for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st
from typing import List, Dict, Any

try:
    from pa_core.validators import (
        ValidationResult,
        validate_correlations,
        validate_capital_allocation,
        validate_simulation_parameters,
        format_validation_messages,
    )
except ImportError:
    # Fallback for development
    ValidationResult = None
    validate_correlations = None
    validate_capital_allocation = None
    validate_simulation_parameters = None
    format_validation_messages = None


def display_validation_results(results: List[ValidationResult], title: str = "Validation Results") -> None:
    """Display validation results in Streamlit with appropriate styling.
    
    Args:
        results: List of validation results to display
        title: Title for the validation section
    """
    if not results:
        st.success("✅ All validations passed!")
        return
    
    # Separate by severity
    errors = [r for r in results if r.severity == 'error' and not r.is_valid]
    warnings = [r for r in results if r.severity == 'warning']
    infos = [r for r in results if r.severity == 'info']
    
    # Display errors
    if errors:
        st.error("❌ **Validation Errors**")
        for error in errors:
            st.error(f"• {error.message}")
            if error.details:
                with st.expander("Details", expanded=False):
                    st.json(error.details)
    
    # Display warnings
    if warnings:
        st.warning("⚠️ **Validation Warnings**")
        for warning in warnings:
            st.warning(f"• {warning.message}")
            if warning.details:
                with st.expander("Details", expanded=False):
                    st.json(warning.details)
    
    # Display info (collapsed by default)
    if infos:
        with st.expander("ℹ️ **Validation Info**", expanded=False):
            for info in infos:
                st.info(f"• {info.message}")
                if info.details:
                    st.json(info.details)


def create_validation_sidebar() -> Dict[str, Any]:
    """Create a sidebar section for validation controls and display.
    
    Returns:
        Dictionary containing validation settings and results
    """
    st.sidebar.subheader("🔍 Validation Settings")
    
    # Validation controls
    validate_on_change = st.sidebar.checkbox("Real-time validation", value=True, help="Validate parameters as you change them")
    show_details = st.sidebar.checkbox("Show validation details", value=False, help="Show detailed validation information")
    show_warnings = st.sidebar.checkbox("Show warnings", value=True, help="Display validation warnings")
    
    return {
        'validate_on_change': validate_on_change,
        'show_details': show_details,
        'show_warnings': show_warnings,
    }


def validate_scenario_config(config_data: Dict[str, Any], validation_settings: Dict[str, Any]) -> List[ValidationResult]:
    """Validate a scenario configuration and return results.
    
    Args:
        config_data: Configuration data to validate
        validation_settings: Validation settings from sidebar
        
    Returns:
        List of validation results
    """
    if not validation_settings.get('validate_on_change', True):
        return []
    
    all_results = []
    
    # Validate correlations if present
    if validate_correlations:
        correlations = {}
        for key in ['rho_idx_H', 'rho_idx_E', 'rho_idx_M', 'rho_H_E', 'rho_H_M', 'rho_E_M']:
            if key in config_data:
                correlations[key] = config_data[key]
        
        if correlations:
            correlation_results = validate_correlations(correlations)
            all_results.extend(correlation_results)
    
    # Validate capital allocation if present
    if validate_capital_allocation and all(
        key in config_data for key in ['external_pa_capital', 'active_ext_capital', 'internal_pa_capital']
    ):
        capital_results = validate_capital_allocation(
            external_pa_capital=config_data.get('external_pa_capital', 0.0),
            active_ext_capital=config_data.get('active_ext_capital', 0.0),
            internal_pa_capital=config_data.get('internal_pa_capital', 0.0),
            total_fund_capital=config_data.get('total_fund_capital', 1000.0),
            reference_sigma=config_data.get('reference_sigma', 0.01),
            volatility_multiple=config_data.get('volatility_multiple', 3.0)
        )
        all_results.extend(capital_results)
    
    # Validate simulation parameters if present
    if validate_simulation_parameters and 'N_SIMULATIONS' in config_data:
        step_sizes = {}
        for key in ['external_step_size_pct', 'in_house_return_step_pct', 'alpha_ext_return_step_pct']:
            if key in config_data:
                step_sizes[key] = config_data[key]
        
        sim_results = validate_simulation_parameters(
            n_simulations=config_data['N_SIMULATIONS'],
            step_sizes=step_sizes if step_sizes else None
        )
        all_results.extend(sim_results)
    
    # Filter based on settings
    if not validation_settings.get('show_warnings', True):
        all_results = [r for r in all_results if r.severity != 'warning']
    
    return all_results


def display_psd_projection_info(psd_info: Dict[str, Any]) -> None:
    """Display PSD projection information in a formatted way.
    
    Args:
        psd_info: PSD projection information from covariance validation
    """
    if not psd_info.get('was_projected', False):
        st.success("✅ Covariance matrix is positive semidefinite")
        return
    
    st.warning("⚠️ **Covariance Matrix Projected to PSD**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Max Delta (Δλmax)", 
            f"{psd_info.get('max_eigenvalue_delta', 0):.2e}",
            help="Maximum change in eigenvalues due to projection"
        )
    
    with col2:
        st.metric(
            "Max Matrix Delta", 
            f"{psd_info.get('max_delta', 0):.2e}",
            help="Maximum absolute change in matrix elements"
        )
    
    with st.expander("Projection Details", expanded=False):
        st.write("**Original min eigenvalue:**", f"{psd_info.get('original_min_eigenvalue', 0):.6f}")
        st.write("**Projected min eigenvalue:**", f"{psd_info.get('projected_min_eigenvalue', 0):.6f}")
        
        st.info(
            "The covariance matrix was not positive semidefinite and has been "
            "projected to the nearest PSD matrix using Higham's method. This "
            "ensures numerical stability in simulations."
        )


def create_margin_buffer_display(
    margin_requirement: float, 
    available_buffer: float, 
    total_capital: float = 1000.0
) -> None:
    """Display margin requirement and buffer status.
    
    Args:
        margin_requirement: Required margin amount
        available_buffer: Available capital buffer
        total_capital: Total capital available
    """
    st.subheader("💰 Capital Buffer Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Margin Requirement", 
            f"${margin_requirement:.1f}M",
            help="Required margin for beta backing"
        )
    
    with col2:
        buffer_ratio = available_buffer / total_capital if total_capital > 0 else 0
        st.metric(
            "Available Buffer", 
            f"${available_buffer:.1f}M",
            f"{buffer_ratio:.1%}",
            help="Capital remaining after all allocations and margin"
        )
    
    with col3:
        utilization = (total_capital - available_buffer) / total_capital if total_capital > 0 else 0
        st.metric(
            "Capital Utilization",
            f"{utilization:.1%}",
            help="Percentage of total capital allocated"
        )
    
    # Visual buffer indicator
    if buffer_ratio < 0:
        st.error("🔴 **Capital Shortfall!** Insufficient capital for current allocation.")
    elif buffer_ratio < 0.1:
        st.warning("🟡 **Low Buffer.** Consider reducing allocations or increasing total capital.")
    elif buffer_ratio < 0.2:
        st.info("🟡 **Moderate Buffer.** Capital allocation is near limits.")
    else:
        st.success("🟢 **Healthy Buffer.** Sufficient capital available.")


def validation_status_indicator(results: List[ValidationResult]) -> str:
    """Return a status indicator emoji based on validation results.
    
    Args:
        results: List of validation results
        
    Returns:
        Status emoji string
    """
    if not results:
        return "✅"
    
    has_errors = any(not r.is_valid for r in results)
    has_warnings = any(r.severity == 'warning' for r in results)
    
    if has_errors:
        return "❌"
    elif has_warnings:
        return "⚠️"
    else:
        return "✅"