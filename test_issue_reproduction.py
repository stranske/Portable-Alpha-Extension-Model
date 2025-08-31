#!/usr/bin/env python3
"""Test script to reproduce the issue with load_yaml_str bypassing validation."""

import sys
from pathlib import Path

# Add the repo to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pa_core.presets import AlphaPreset, PresetLibrary
import yaml

def test_current_behavior():
    """Test the current behavior to understand the issue."""
    lib = PresetLibrary()
    
    # Test case 1: Valid YAML where preset.id matches the dictionary key
    valid_yaml = """
preset_a:
  id: preset_a
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset_b:
  id: preset_b
  mu: 0.15
  sigma: 0.25
  rho: 0.35
"""
    
    print("=== Test 1: Valid YAML ===")
    try:
        lib.load_yaml_str(valid_yaml)
        print("✓ Valid YAML loaded successfully")
        print(f"  Loaded presets: {list(lib.presets.keys())}")
    except Exception as e:
        print(f"✗ Unexpected error with valid YAML: {e}")
    
    # Test case 2: YAML where preset.id does NOT match the dictionary key
    # This is the potential issue - if validation logic has a bug, this could overwrite silently
    mismatched_yaml = """
preset_a:
  id: different_id  # This should be preset_a but isn't
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset_b:
  id: preset_b
  mu: 0.15
  sigma: 0.25
  rho: 0.35
"""
    
    print("\n=== Test 2: Mismatched ID YAML ===")
    lib = PresetLibrary()  # Fresh library
    try:
        lib.load_yaml_str(mismatched_yaml)
        print("✓ Mismatched YAML loaded (this might be the issue)")
        print(f"  Loaded presets: {list(lib.presets.keys())}")
        # Check if the preset ID actually matches what we expect
        for key, preset in lib.presets.items():
            print(f"  Key: {key}, Preset ID: {preset.id}")
    except Exception as e:
        print(f"✗ Error with mismatched YAML: {e}")
    
    # Test case 3: Duplicate IDs within the YAML (should fail)
    duplicate_yaml = """
preset_a:
  id: same_id
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset_b:
  id: same_id  # Duplicate ID
  mu: 0.15
  sigma: 0.25
  rho: 0.35
"""
    
    print("\n=== Test 3: Duplicate ID YAML ===")
    lib = PresetLibrary()  # Fresh library
    try:
        lib.load_yaml_str(duplicate_yaml)
        print("✗ Duplicate YAML loaded (should have failed)")
        print(f"  Loaded presets: {list(lib.presets.keys())}")
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate YAML: {e}")
    except Exception as e:
        print(f"✗ Unexpected error with duplicate YAML: {e}")

def test_json_behavior():
    """Test JSON loading which doesn't have the same validation."""
    lib = PresetLibrary()
    
    # Test case 1: Valid JSON
    valid_json = """{
  "preset_a": {
    "id": "preset_a",
    "mu": 0.1,
    "sigma": 0.2,
    "rho": 0.3
  },
  "preset_b": {
    "id": "preset_b",
    "mu": 0.15,
    "sigma": 0.25,
    "rho": 0.35
  }
}"""
    
    print("\n=== Test 4: Valid JSON ===")
    try:
        lib.load_json_str(valid_json)
        print("✓ Valid JSON loaded successfully")
        print(f"  Loaded presets: {list(lib.presets.keys())}")
    except Exception as e:
        print(f"✗ Unexpected error with valid JSON: {e}")
    
    # Test case 2: JSON with duplicate IDs (load_json_str has no validation)
    duplicate_json = """{
  "preset_a": {
    "id": "same_id",
    "mu": 0.1,
    "sigma": 0.2,
    "rho": 0.3
  },
  "preset_b": {
    "id": "same_id",
    "mu": 0.15,
    "sigma": 0.25,
    "rho": 0.35
  }
}"""
    
    print("\n=== Test 5: Duplicate ID JSON ===")
    lib = PresetLibrary()  # Fresh library
    try:
        lib.load_json_str(duplicate_json)
        print("✗ Duplicate JSON loaded (issue: no validation in load_json_str)")
        print(f"  Loaded presets: {list(lib.presets.keys())}")
        for key, preset in lib.presets.items():
            print(f"  Key: {key}, Preset ID: {preset.id}")
    except Exception as e:
        print(f"✓ Correctly rejected duplicate JSON: {e}")

if __name__ == "__main__":
    test_current_behavior()
    test_json_behavior()