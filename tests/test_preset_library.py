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

    lib_yaml.load_yaml_str(lib.to_yaml_str())
    lib_json.load_json_str(lib.to_json_str())
    assert lib_yaml.to_dict() == lib.to_dict()
    assert lib_json.to_dict() == lib.to_dict()


def test_load_yaml_str_duplicate_validation():
    """Test that load_yaml_str properly validates duplicate IDs."""
    lib = PresetLibrary()

    # Test duplicate IDs within YAML should fail
    duplicate_yaml = """
preset_a:
  id: same_id
  mu: 0.1
  sigma: 0.2
  rho: 0.3
preset_b:
  id: same_id
  mu: 0.15
  sigma: 0.25
  rho: 0.35
"""

    with pytest.raises(ValueError, match="Duplicate preset IDs"):
        lib.load_yaml_str(duplicate_yaml)


def test_load_yaml_str_id_mismatch_validation():
    """Test that load_yaml_str validates that preset.id matches the dictionary key."""
    lib = PresetLibrary()

    # Test mismatched ID should fail
    mismatched_yaml = """
preset_a:
  id: different_id
  mu: 0.1
  sigma: 0.2
  rho: 0.3
"""

    with pytest.raises(
        ValueError, match="Preset ID 'different_id' does not match its key 'preset_a'"
    ):
        lib.load_yaml_str(mismatched_yaml)


def test_load_json_str_duplicate_validation():
    """Test that load_json_str rejects duplicate IDs before key mismatches."""
    lib = PresetLibrary()

    duplicate_json = """{
  "same_id": {
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

    with pytest.raises(ValueError, match="Duplicate preset IDs"):
        lib.load_json_str(duplicate_json)


def test_load_json_str_id_mismatch_validation():
    """Test that load_json_str validates that preset.id matches the dictionary key."""
    lib = PresetLibrary()

    # Test mismatched ID should fail
    mismatched_json = """{
  "preset_a": {
    "id": "different_id",
    "mu": 0.1,
    "sigma": 0.2,
    "rho": 0.3
  }
}"""

    with pytest.raises(
        ValueError, match="Preset ID 'different_id' does not match its key 'preset_a'"
    ):
        lib.load_json_str(mismatched_json)
