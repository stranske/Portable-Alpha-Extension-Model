"""Unit tests for validation functions."""

from __future__ import annotations

import numpy as np
import pytest

from pa_core.validators import (
    ValidationResult,
    PSDProjectionInfo,
    validate_correlations,
    validate_covariance_matrix_psd,
    validate_capital_allocation,
    validate_simulation_parameters,
    calculate_margin_requirement,
    format_validation_messages,
)


class TestCorrelationValidation:
    """Test correlation validation functions."""
    
    def test_valid_correlations(self):
        """Test validation with all valid correlations."""
        correlations = {
            "rho_idx_H": 0.05,
            "rho_idx_E": 0.0,
            "rho_H_E": 0.1,
        }
        results = validate_correlations(correlations)
        
        # Should have no error results  
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0
    
    def test_invalid_correlations(self):
        """Test validation with invalid correlations."""
        correlations = {
            "rho_idx_H": 1.5,  # Too high
            "rho_idx_E": -1.2,  # Too low
            "rho_H_E": 0.1,  # Valid
        }
        results = validate_correlations(correlations)
        
        # Should have 2 error results
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 2
        
        # Check error messages contain the parameter names
        error_messages = [r.message for r in errors]
        assert any("rho_idx_H" in msg for msg in error_messages)
        assert any("rho_idx_E" in msg for msg in error_messages)
    
    def test_high_correlation_warning(self):
        """Test warning for high correlations."""
        correlations = {"rho_test": 0.97}  # High but valid
        results = validate_correlations(correlations)
        
        # Should pass validation but have warning
        assert all(r.is_valid for r in results)
        warnings = [r for r in results if r.severity == 'warning']
        assert len(warnings) == 1
        assert "High correlation" in warnings[0].message


class TestCovarianceMatrixValidation:
    """Test covariance matrix PSD validation."""
    
    def test_psd_matrix(self):
        """Test validation with already PSD matrix."""
        # Simple 2x2 PSD matrix
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        result, psd_info = validate_covariance_matrix_psd(cov)
        
        assert result.is_valid
        assert not psd_info.was_projected
        assert psd_info.max_delta == 0.0
    
    def test_non_psd_matrix(self):
        """Test validation with non-PSD matrix requiring projection."""
        # Non-PSD matrix (negative eigenvalue)
        cov = np.array([[1.0, 1.1], [1.1, 1.0]])  # Determinant < 0
        result, psd_info = validate_covariance_matrix_psd(cov)
        
        assert result.is_valid  # Still valid after projection
        assert result.severity == 'warning'
        assert psd_info.was_projected
        assert psd_info.max_delta > 0
        assert "Projected to PSD" in result.message
    
    def test_psd_projection_details(self):
        """Test that PSD projection provides detailed information."""
        # Construct a matrix that will need projection
        cov = np.array([[1.0, 0.99], [0.99, 0.5]])  # Will have small negative eigenvalue
        result, psd_info = validate_covariance_matrix_psd(cov)
        
        # Check that detailed info is provided
        assert isinstance(psd_info.original_min_eigenvalue, float)
        assert isinstance(psd_info.projected_min_eigenvalue, float)
        assert isinstance(psd_info.max_eigenvalue_delta, float)
        
        if psd_info.was_projected:
            assert psd_info.projected_min_eigenvalue >= -1e-10  # Should be non-negative


class TestCapitalAllocationValidation:
    """Test capital allocation validation."""
    
    def test_valid_capital_allocation(self):
        """Test validation with valid capital allocation."""
        results = validate_capital_allocation(
            external_pa_capital=200.0,
            active_ext_capital=300.0,
            internal_pa_capital=400.0,
            total_fund_capital=1000.0,
            reference_sigma=0.01,
            volatility_multiple=2.0
        )
        
        # Should have no errors
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0
        
        # Should have buffer info
        info_results = [r for r in results if r.severity == 'info']
        assert len(info_results) > 0
    
    def test_capital_allocation_exceeds_total(self):
        """Test validation when capital allocation exceeds total."""
        results = validate_capital_allocation(
            external_pa_capital=500.0,
            active_ext_capital=400.0,
            internal_pa_capital=300.0,  # Total = 1200 > 1000
            total_fund_capital=1000.0,
        )
        
        # Should have error
        errors = [r for r in results if not r.is_valid]
        assert len(errors) >= 1
        assert any("exceeds total fund capital" in r.message for r in errors)
    
    def test_margin_plus_internal_exceeds_total(self):
        """Test validation when margin + internal PA exceeds total."""
        results = validate_capital_allocation(
            external_pa_capital=100.0,
            active_ext_capital=100.0,
            internal_pa_capital=800.0,  # High internal PA
            total_fund_capital=1000.0,
            reference_sigma=0.05,  # High volatility
            volatility_multiple=5.0  # High multiplier -> high margin
        )
        
        # Should have error about margin + internal exceeding total
        errors = [r for r in results if not r.is_valid]
        assert len(errors) >= 1
        assert any("Margin requirement" in r.message and "exceeds total capital" in r.message for r in errors)
    
    def test_low_buffer_warning(self):
        """Test warning when capital buffer is low."""
        # Set up scenario with low buffer
        # margin = 0.02 * 2.5 * 1000 = 50M
        # internal_pa = 900M  
        # margin + internal_pa = 950M, leaving 50M buffer (5%), which should be low
        results = validate_capital_allocation(
            external_pa_capital=25.0,
            active_ext_capital=25.0, 
            internal_pa_capital=900.0,  # Very high internal PA to leave small buffer
            total_fund_capital=1000.0,
            reference_sigma=0.02,
            volatility_multiple=2.5
        )
        
        # Should have warning about low buffer (< 10% threshold)
        warning_results = [r for r in results if r.severity == 'warning']
        buffer_warnings = [w for w in warning_results if "Low capital buffer" in w.message]
        assert len(buffer_warnings) >= 1
    
    def test_margin_requirement_calculation(self):
        """Test margin requirement calculation."""
        margin = calculate_margin_requirement(
            reference_sigma=0.02,
            volatility_multiple=3.0,
            total_capital=1000.0
        )
        expected = 0.02 * 3.0 * 1000.0  # 60.0
        assert margin == expected


class TestSimulationParameterValidation:
    """Test simulation parameter validation."""
    
    def test_valid_simulation_params(self):
        """Test validation with valid simulation parameters."""
        results = validate_simulation_parameters(
            n_simulations=5000,
            step_sizes={'param1': 0.5, 'param2': 1.0}
        )
        
        # Should have no errors
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0
    
    def test_low_n_simulations_error(self):
        """Test error for too low N_SIMULATIONS."""
        results = validate_simulation_parameters(n_simulations=50)
        
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 1
        assert "too low" in errors[0].message
        assert "50" in errors[0].message
    
    def test_low_n_simulations_warning(self):
        """Test warning for low but acceptable N_SIMULATIONS."""
        results = validate_simulation_parameters(n_simulations=500)
        
        # Should pass but have warning
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 0
        
        warnings = [r for r in results if r.severity == 'warning']
        assert len(warnings) == 1
        assert "below recommended" in warnings[0].message
    
    def test_high_n_simulations_warning(self):
        """Test warning for very high N_SIMULATIONS."""
        results = validate_simulation_parameters(n_simulations=200000)
        
        warnings = [r for r in results if r.severity == 'warning']
        assert len(warnings) == 1
        assert "very high" in warnings[0].message
    
    def test_invalid_step_size(self):
        """Test error for invalid step sizes."""
        results = validate_simulation_parameters(
            n_simulations=1000,
            step_sizes={'param1': 0.0, 'param2': -0.5}
        )
        
        errors = [r for r in results if not r.is_valid]
        assert len(errors) == 2  # Both invalid step sizes
        
        error_messages = [r.message for r in errors]
        assert any("param1" in msg and "must be positive" in msg for msg in error_messages)
        assert any("param2" in msg and "must be positive" in msg for msg in error_messages)
    
    def test_small_step_size_warning(self):
        """Test warning for very small step sizes."""
        results = validate_simulation_parameters(
            n_simulations=1000,
            step_sizes={'param1': 0.01}  # Very small
        )
        
        warnings = [r for r in results if r.severity == 'warning']
        assert len(warnings) == 1
        assert "Very small step size" in warnings[0].message


class TestValidationFormatting:
    """Test validation message formatting."""
    
    def test_format_empty_results(self):
        """Test formatting with no results."""
        message = format_validation_messages([])
        assert message == "All validations passed."
    
    def test_format_mixed_results(self):
        """Test formatting with mixed severity results."""
        results = [
            ValidationResult(is_valid=False, message="Error 1", severity='error'),
            ValidationResult(is_valid=True, message="Warning 1", severity='warning'),
            ValidationResult(is_valid=True, message="Info 1", severity='info'),
        ]
        
        message = format_validation_messages(results)
        
        assert "❌ ERRORS:" in message
        assert "⚠️ WARNINGS:" in message
        assert "ℹ️ INFO:" in message
        assert "Error 1" in message
        assert "Warning 1" in message
        assert "Info 1" in message
    
    def test_format_with_details(self):
        """Test formatting with details included."""
        results = [
            ValidationResult(
                is_valid=False, 
                message="Test error", 
                severity='error',
                details={'param': 'value', 'context': 'test'}
            )
        ]
        
        message = format_validation_messages(results, include_details=True)
        assert "Details:" in message
        assert "param" in message
        assert "value" in message