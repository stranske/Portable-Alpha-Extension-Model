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
    """Test that load_json_str fails gracefully when duplicate IDs exist."""
    lib = PresetLibrary()
    # Add some existing presets
    lib.add(AlphaPreset(id="existing", mu=0.1, sigma=0.1, rho=0.1))
    
    # Create JSON data with duplicate IDs
    json_data = """{
        "preset1": {"id": "A", "mu": 0.1, "sigma": 0.2, "rho": 0.3},
        "preset2": {"id": "A", "mu": 0.2, "sigma": 0.3, "rho": 0.4}
    }"""
    
    # This should currently cause a partial loading failure
    with pytest.raises(ValueError, match="preset 'A' already exists"):
        lib.load_json_str(json_data)
    
    # After the failure, the library should be in an inconsistent state:
    # - existing presets are cleared
    # - only the first duplicate was added
    # - the original presets are lost
    assert "existing" not in lib.presets  # This shows the problem!
    assert len(lib.presets) == 1  # Only first duplicate was added


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