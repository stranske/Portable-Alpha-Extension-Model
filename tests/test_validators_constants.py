"""Unit tests for validation functions with named constants."""

from __future__ import annotations

import pytest

# Import with proper error handling since file might not exist in all branches
try:
    from pa_core.validators import (
        LOW_BUFFER_THRESHOLD,
        MIN_RECOMMENDED_STEP_SIZE,
        NUMERICAL_STABILITY_EPSILON,
        SYNTHETIC_DATA_MEAN,
        SYNTHETIC_DATA_STD,
        TEST_TOLERANCE_EPSILON,
        ValidationResult,
        validate_simulation_parameters,
    )
except ImportError:
    pytest.skip("Validators module not available", allow_module_level=True)


class TestValidationConstants:
    """Test that validation constants are properly defined and used."""

    def test_min_recommended_step_size_constant_defined(self):
        """Test that MIN_RECOMMENDED_STEP_SIZE is defined."""
        assert MIN_RECOMMENDED_STEP_SIZE == 0.1
        assert isinstance(MIN_RECOMMENDED_STEP_SIZE, float)

    def test_low_buffer_threshold_constant_defined(self):
        """Test that LOW_BUFFER_THRESHOLD is defined."""
        assert LOW_BUFFER_THRESHOLD == 0.1
        assert isinstance(LOW_BUFFER_THRESHOLD, float)

    def test_step_size_validation_uses_constant(self):
        """Test that step size validation uses the named constant."""
        # Test step size exactly at the threshold
        results = validate_simulation_parameters(
            n_simulations=1000, step_sizes={"test_param": MIN_RECOMMENDED_STEP_SIZE}
        )

        # Should not trigger warning at exactly the threshold
        warnings = [
            r
            for r in results
            if r.severity == "warning" and "Very small step size" in r.message
        ]
        assert len(warnings) == 0

    def test_step_size_below_threshold_triggers_warning(self):
        """Test that step size below threshold triggers warning with constant value."""
        small_step = MIN_RECOMMENDED_STEP_SIZE - 0.01  # Just below threshold
        results = validate_simulation_parameters(
            n_simulations=1000, step_sizes={"test_param": small_step}
        )

        # Should trigger warning
        warnings = [
            r
            for r in results
            if r.severity == "warning" and "Very small step size" in r.message
        ]
        assert len(warnings) == 1

        # Check that the warning includes the correct constant value
        warning = warnings[0]
        assert warning.details["minimum_recommended"] == MIN_RECOMMENDED_STEP_SIZE
        assert warning.details["value"] == small_step

    def test_step_size_above_threshold_no_warning(self):
        """Test that step size above threshold doesn't trigger warning."""
        large_step = MIN_RECOMMENDED_STEP_SIZE + 0.1  # Well above threshold
        results = validate_simulation_parameters(
            n_simulations=1000, step_sizes={"test_param": large_step}
        )

        # Should not trigger step size warning
        step_warnings = [
            r
            for r in results
            if r.severity == "warning" and "Very small step size" in r.message
        ]
        assert len(step_warnings) == 0

    def test_multiple_step_sizes_mixed_validation(self):
        """Test validation with multiple step sizes, some below threshold."""
        results = validate_simulation_parameters(
            n_simulations=1000,
            step_sizes={
                "small_param": MIN_RECOMMENDED_STEP_SIZE - 0.01,  # Below threshold
                "good_param": MIN_RECOMMENDED_STEP_SIZE + 0.1,  # Above threshold
                "boundary_param": MIN_RECOMMENDED_STEP_SIZE,  # At threshold
            },
        )

        # Should only warn about the small step size
        step_warnings = [
            r
            for r in results
            if r.severity == "warning" and "Very small step size" in r.message
        ]
        assert len(step_warnings) == 1
        assert "small_param" in step_warnings[0].message

    def test_constant_consistency(self):
        """Test that constants are used consistently throughout the module."""
        # This test ensures that if someone accidentally uses a hardcoded value,
        # it will be caught. We test the actual function behavior matches the constant.

        # Test with value slightly below the constant
        results = validate_simulation_parameters(
            n_simulations=1000, step_sizes={"param": MIN_RECOMMENDED_STEP_SIZE - 0.001}
        )

        warnings = [
            r
            for r in results
            if r.severity == "warning" and "Very small step size" in r.message
        ]
        assert len(warnings) == 1

        # The details should contain the exact constant value
        assert warnings[0].details["minimum_recommended"] == MIN_RECOMMENDED_STEP_SIZE

    def test_numerical_stability_epsilon_constant(self):
        """Test that NUMERICAL_STABILITY_EPSILON constant is defined correctly."""
        # Test that the constant exists and has the expected value
        assert NUMERICAL_STABILITY_EPSILON == 1e-12
        assert isinstance(NUMERICAL_STABILITY_EPSILON, float)

    def test_test_tolerance_epsilon_constant(self):
        """Test that TEST_TOLERANCE_EPSILON constant is defined correctly."""
        # Test that the constant exists and has the expected value
        assert TEST_TOLERANCE_EPSILON == 1e-12
        assert isinstance(TEST_TOLERANCE_EPSILON, float)

    def test_epsilon_constants_are_small_positive_values(self):
        """Test that epsilon constants are appropriately small positive values."""
        # Both epsilon constants should be very small positive numbers
        assert NUMERICAL_STABILITY_EPSILON == 1e-12
        assert TEST_TOLERANCE_EPSILON == 1e-12

    def test_synthetic_data_constants_defined(self):
        """Test that synthetic data constants are defined correctly."""
        # Test that both constants exist and have the expected values
        assert SYNTHETIC_DATA_MEAN == 0.0
        assert SYNTHETIC_DATA_STD == 0.01
        assert isinstance(SYNTHETIC_DATA_MEAN, float)
        assert isinstance(SYNTHETIC_DATA_STD, float)

    def test_synthetic_data_constants_are_realistic(self):
        """Test that synthetic data constants represent realistic values."""
        # Mean should be zero (neutral expected return)
        assert SYNTHETIC_DATA_MEAN == 0.0
        
        # Standard deviation should be positive and reasonable for test data
        assert SYNTHETIC_DATA_STD > 0.0
        assert SYNTHETIC_DATA_STD <= 1.0  # Should be reasonable for test scenarios
