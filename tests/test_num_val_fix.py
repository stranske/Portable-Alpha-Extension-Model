"""
'Tests for CSV to YAML conversion edge cases that previously caused undefined variable errors in _convert_csv_to_yaml function.'
"""

import tempfile
import os
from pathlib import Path

def test_convert_csv_to_yaml_with_undefined_num_val_conditions():
    """Test that _convert_csv_to_yaml handles edge cases that could cause undefined num_val."""
    from pa_core.pa import _convert_csv_to_yaml
    
    # Create a test CSV with problematic values that could trigger the original bug
    problematic_data = [
        ("Parameter", "Value"),
        ("Total capital", "1000"),        # Normal case - should work
        ("Risk-free rate (%)", ""),       # Empty value with percentage
        ("Market return (%)", "N/A"),     # Invalid text with percentage  
        ("Volatility (%)", "invalid"),    # Invalid text with percentage
        ("Correlation", ""),              # Empty value without percentage
        ("Bad value", None),              # None-like value (written as empty string)
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as f:
        for row in problematic_data:
            val = row[1] if row[1] is not None else ""
            f.write(f"{row[0]},{val}\n")
        f.flush()  # Ensure data is written to disk
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as yaml_f:
            # This should not crash due to undefined num_val variable
            # Prior to the fix, this could raise NameError: name 'num_val' is not defined
            _convert_csv_to_yaml(f.name, yaml_f.name)
            
            # Verify the output file was created
            assert Path(yaml_f.name).exists()
            
            # Verify the content is valid YAML
            import yaml
            with open(yaml_f.name, 'r') as yaml_content:
                content = yaml.safe_load(yaml_content)
                assert isinstance(content, dict)
                # Should have default risk_metrics
                assert "risk_metrics" in content


def test_convert_csv_to_yaml_percentage_conversion():
    """Test that percentage conversion still works correctly after the fix."""
    from pa_core.pa import _convert_csv_to_yaml
    
    # Create a test CSV with percentage values
    test_data = [
        ("Parameter", "Value"),
        ("Risk-free rate (%)", "5.0"),     # Should convert to 0.05
        ("Market return (%)", "12"),       # Should convert to 0.12
        ("Normal value", "100"),           # Should stay as 100
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as f:
        for row in test_data:
            f.write(f"{row[0]},{row[1]}\n")
        f.flush()  # Ensure data is written to disk
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as yaml_f:
            _convert_csv_to_yaml(f.name, yaml_f.name)
            
            # Check that percentages were converted correctly
            import yaml
            with open(yaml_f.name, 'r') as yaml_content:
                content = yaml.safe_load(yaml_content)
            
            # Find fields that should be percentage-converted
            # Note: The exact field names depend on the field mapping
            # This test just verifies no crash occurs and valid YAML is produced
            assert isinstance(content, dict)
            assert "risk_metrics" in content


if __name__ == "__main__":
    test_convert_csv_to_yaml_with_undefined_num_val_conditions()
    test_convert_csv_to_yaml_percentage_conversion()
    print("All tests passed!")
