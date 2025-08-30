from pathlib import Path
import sys

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from pa_core.presets import AlphaPreset, PresetLibrary


def test_preset_library_crud(tmp_path):
    lib = PresetLibrary()
    p = AlphaPreset(id="A", mu=0.1, sigma=0.2, rho=0.3)
    lib.add(p)
    assert lib.get("A").mu == 0.1
    lib.update(AlphaPreset(id="A", mu=0.2, sigma=0.2, rho=0.3))
    assert lib.get("A").mu == 0.2
    lib.delete("A")
    assert "A" not in lib.presets


def test_preset_import_export(tmp_path):
    lib = PresetLibrary([AlphaPreset(id="A", mu=0.1, sigma=0.2, rho=0.3)])
    yaml_path = tmp_path / "presets.yaml"
    json_path = tmp_path / "presets.json"
    lib.to_yaml(yaml_path)
    lib.to_json(json_path)
    lib_yaml = PresetLibrary.from_yaml(yaml_path)
    lib_json = PresetLibrary.from_json(json_path)
    assert lib_yaml.get("A").rho == 0.3
    assert lib_json.get("A").sigma == 0.2


def test_load_json_str_with_duplicates():
    """Test that load_json_str handles duplicates correctly with validation."""
    lib = PresetLibrary()
    # Add some existing presets
    lib.add(AlphaPreset(id="existing", mu=0.1, sigma=0.1, rho=0.1))
    
    # Create JSON data with duplicate IDs
    json_data = """{
        "preset1": {"id": "A", "mu": 0.1, "sigma": 0.2, "rho": 0.3},
        "preset2": {"id": "A", "mu": 0.2, "sigma": 0.3, "rho": 0.4}
    }"""
    
    # This should fail with validation error BEFORE clearing existing presets
    with pytest.raises(ValueError, match="Duplicate preset IDs found in input: A"):
        lib.load_json_str(json_data)
    
    # After the failure, existing presets should still be there
    assert "existing" in lib.presets  # Fixed!
    assert len(lib.presets) == 1


def test_load_yaml_str_with_duplicates():
    """Test that load_yaml_str handles duplicates correctly with validation."""
    lib = PresetLibrary()
    # Add some existing presets
    lib.add(AlphaPreset(id="existing", mu=0.1, sigma=0.1, rho=0.1))
    
    # Create YAML data with duplicate IDs
    yaml_data = """
preset1:
  id: A
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset2:
  id: A
  mu: 0.2
  sigma: 0.3
  rho: 0.4
"""
    
    # This should fail with validation error BEFORE clearing existing presets
    with pytest.raises(ValueError, match="Duplicate preset IDs found in input: A"):
        lib.load_yaml_str(yaml_data)
    
    # After the failure, existing presets should still be there
    assert "existing" in lib.presets  # This is good!
    assert len(lib.presets) == 1


def test_load_yaml_str_successful():
    """Test successful loading of YAML data without duplicates."""
    lib = PresetLibrary()
    lib.add(AlphaPreset(id="existing", mu=0.1, sigma=0.1, rho=0.1))
    
    yaml_data = """
preset1:
  id: A
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset2:
  id: B
  mu: 0.2
  sigma: 0.3
  rho: 0.4
"""
    
    lib.load_yaml_str(yaml_data)
    
    # Should replace all existing presets with new ones
    assert "existing" not in lib.presets
    assert len(lib.presets) == 2
    assert lib.get("A").mu == 0.1
    assert lib.get("B").sigma == 0.3


def test_load_json_str_successful():
    """Test successful loading of JSON data without duplicates."""
    lib = PresetLibrary()
    lib.add(AlphaPreset(id="existing", mu=0.1, sigma=0.1, rho=0.1))
    
    json_data = """{
        "preset1": {"id": "A", "mu": 0.1, "sigma": 0.2, "rho": 0.3},
        "preset2": {"id": "B", "mu": 0.2, "sigma": 0.3, "rho": 0.4}
    }"""
    
    lib.load_json_str(json_data)
    
    # Should replace all existing presets with new ones
    assert "existing" not in lib.presets
    assert len(lib.presets) == 2
    assert lib.get("A").mu == 0.1
    assert lib.get("B").sigma == 0.3


def test_partial_loading_failure_prevention():
    """Test that loading failures don't leave the library in an inconsistent state."""
    lib = PresetLibrary()
    lib.add(AlphaPreset(id="important", mu=0.5, sigma=0.1, rho=0.2))
    lib.add(AlphaPreset(id="critical", mu=0.3, sigma=0.2, rho=0.4))
    
    # Store original state
    original_count = len(lib.presets)
    original_important = lib.get("important")
    original_critical = lib.get("critical")
    
    # Try to load invalid data with duplicates
    bad_json = """{
        "preset1": {"id": "new1", "mu": 0.1, "sigma": 0.1, "rho": 0.1},
        "preset2": {"id": "duplicate", "mu": 0.2, "sigma": 0.2, "rho": 0.2},
        "preset3": {"id": "duplicate", "mu": 0.3, "sigma": 0.3, "rho": 0.3}
    }"""
    
    bad_yaml = """
preset1:
  id: new1
  mu: 0.1
  sigma: 0.1
  rho: 0.1
preset2:
  id: duplicate
  mu: 0.2
  sigma: 0.2
  rho: 0.2
preset3:
  id: duplicate
  mu: 0.3
  sigma: 0.3
  rho: 0.3
"""
    
    # Both should fail with validation error and preserve original state
    with pytest.raises(ValueError, match="Duplicate preset IDs found in input: duplicate"):
        lib.load_json_str(bad_json)
    
    # Original state should be preserved
    assert len(lib.presets) == original_count
    assert lib.get("important").mu == original_important.mu
    assert lib.get("critical").sigma == original_critical.sigma
    
    # Try the same with YAML
    with pytest.raises(ValueError, match="Duplicate preset IDs found in input: duplicate"):
        lib.load_yaml_str(bad_yaml)
    
    # Original state should still be preserved
    assert len(lib.presets) == original_count
    assert lib.get("important").rho == original_important.rho
    assert lib.get("critical").mu == original_critical.mu