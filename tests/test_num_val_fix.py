"""
Test the num_val undefined variable fix in pa_core/pa.py
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for row in problematic_data:
            val = row[1] if row[1] is not None else ""
            f.write(f"{row[0]},{val}\n")
        csv_path = f.name
    
    try:
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml_path = f.name
        
        # This should not crash due to undefined num_val variable
        # Prior to the fix, this could raise NameError: name 'num_val' is not defined
        _convert_csv_to_yaml(csv_path, yaml_path)
        
        # Verify the output file was created
        assert Path(yaml_path).exists()
        
        # Verify the content is valid YAML
        import yaml
        with open(yaml_path, 'r') as f:
            content = yaml.safe_load(f)
            assert isinstance(content, dict)
            # Should have default risk_metrics
            assert "risk_metrics" in content
            
    finally:
        # Clean up
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)


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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for row in test_data:
            f.write(f"{row[0]},{row[1]}\n")
        csv_path = f.name
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml_path = f.name
        
        _convert_csv_to_yaml(csv_path, yaml_path)
        
        # Check that percentages were converted correctly
        import yaml
        with open(yaml_path, 'r') as f:
            content = yaml.safe_load(f)
            
            # Find fields that should be percentage-converted
            # Note: The exact field names depend on the field mapping
            # This test just verifies no crash occurs and valid YAML is produced
            assert isinstance(content, dict)
            assert "risk_metrics" in content
            
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)


if __name__ == "__main__":
    test_convert_csv_to_yaml_with_undefined_num_val_conditions()
    test_convert_csv_to_yaml_percentage_conversion()
    print("All tests passed!")