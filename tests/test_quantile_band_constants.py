"""Unit tests for quantile band visualization constants."""

from __future__ import annotations

import pytest
import numpy as np

# Import with proper error handling since file might not exist in all branches
try:
    from pa_core.viz.quantile_band import (
        make,
        DEFAULT_LOWER_QUANTILE,
        DEFAULT_UPPER_QUANTILE,
    )
except ImportError:
    pytest.skip("Quantile band module not available", allow_module_level=True)


class TestQuantileBandConstants:
    """Test that quantile band constants are properly defined and used."""
    
    def test_default_lower_quantile_constant_defined(self):
        """Test that DEFAULT_LOWER_QUANTILE is defined with correct value."""
        assert DEFAULT_LOWER_QUANTILE == 0.1
        assert isinstance(DEFAULT_LOWER_QUANTILE, float)
    
    def test_default_upper_quantile_constant_defined(self):
        """Test that DEFAULT_UPPER_QUANTILE is defined with correct value."""
        assert DEFAULT_UPPER_QUANTILE == 0.9
        assert isinstance(DEFAULT_UPPER_QUANTILE, float)
    
    def test_constants_create_80_percent_confidence_interval(self):
        """Test that the constants represent an 80% confidence interval."""
        # The difference should be 0.8, representing 80% of the distribution
        confidence_interval_width = DEFAULT_UPPER_QUANTILE - DEFAULT_LOWER_QUANTILE
        assert abs(confidence_interval_width - 0.8) < 1e-10
        
        # Each tail excludes 10% (0.1), so total exclusion is 20%
        lower_tail_exclusion = DEFAULT_LOWER_QUANTILE
        upper_tail_exclusion = 1.0 - DEFAULT_UPPER_QUANTILE
        assert abs(lower_tail_exclusion - 0.1) < 1e-10
        assert abs(upper_tail_exclusion - 0.1) < 1e-10
        assert abs(lower_tail_exclusion + upper_tail_exclusion - 0.2) < 1e-10  # 20% total exclusion
    
    def test_quantile_band_uses_default_constants(self):
        """Test that the make function uses the named constants by default."""
        # Create sample data
        sample_data = np.random.normal(0, 0.1, size=(10, 12))
        
        # Call function without specifying quantiles - should use defaults
        fig = make(sample_data)
        
        # Verify the function runs successfully with default constants
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # The function should produce 3 traces: upper bound, lower bound with fill, median
        assert len(fig.data) == 3
    
    def test_quantile_band_accepts_custom_quantiles(self):
        """Test that custom quantiles can still be provided."""
        # Create sample data
        sample_data = np.random.normal(0, 0.1, size=(10, 12))
        
        # Call function with custom quantiles
        custom_quantiles = (0.05, 0.95)  # 90% confidence interval
        fig = make(sample_data, quantiles=custom_quantiles)
        
        # Verify the function runs successfully with custom quantiles
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) == 3
    
    def test_constant_values_are_valid_quantiles(self):
        """Test that the constant values are valid quantile thresholds."""
        # Quantiles must be between 0 and 1
        assert 0 <= DEFAULT_LOWER_QUANTILE <= 1
        assert 0 <= DEFAULT_UPPER_QUANTILE <= 1
        
        # Lower quantile must be less than upper quantile
        assert DEFAULT_LOWER_QUANTILE < DEFAULT_UPPER_QUANTILE
    
    def test_constants_match_documented_tolerance(self):
        """Test that constants match the 10% tolerance mentioned in documentation."""
        # Upper quantile represents 90th percentile = 10% tolerance on upper tail
        assert DEFAULT_UPPER_QUANTILE == 0.9
        
        # Lower quantile represents 10th percentile = 10% tolerance on lower tail  
        assert DEFAULT_LOWER_QUANTILE == 0.1
        
        # The combination provides 10% tolerance on each side
        upper_tail_tolerance = 1.0 - DEFAULT_UPPER_QUANTILE
        lower_tail_tolerance = DEFAULT_LOWER_QUANTILE
        assert abs(upper_tail_tolerance - 0.1) < 1e-10  # 10% tolerance
        assert abs(lower_tail_tolerance - 0.1) < 1e-10  # 10% tolerance
    
    def test_constants_used_in_function_signature(self):
        """Test that the constants are actually used in the function signature."""
        import inspect
        
        # Get the function signature
        sig = inspect.signature(make)
        quantiles_param = sig.parameters['quantiles']
        
        # The default should be a tuple containing our constants
        default_value = quantiles_param.default
        assert isinstance(default_value, tuple)
        assert len(default_value) == 2
        
        # Check that the default values match our constants
        # Note: We can't directly compare the objects since they might be 
        # evaluated expressions, so we compare the actual values
        default_lower, default_upper = default_value
        assert default_lower == DEFAULT_LOWER_QUANTILE
        assert default_upper == DEFAULT_UPPER_QUANTILE