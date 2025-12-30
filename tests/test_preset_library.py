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
    """Test that load_json_str properly validates duplicate IDs through add() method."""
    lib = PresetLibrary()

    # First, let's test that the new validation catches ID mismatches
    mismatched_duplicate_json = """{
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

    # This should fail because same_id != preset_b
    with pytest.raises(ValueError, match="Preset ID 'same_id' does not match its key 'preset_b'"):
        lib.load_json_str(mismatched_duplicate_json)


def test_load_json_str_add_method_duplicate_validation():
    """Test that load_json_str relies on add() method for duplicate validation when keys match IDs."""
    lib = PresetLibrary()

    # Add a preset first
    lib.add(AlphaPreset(id="existing_preset", mu=0.1, sigma=0.2, rho=0.3))

    # Try to load JSON with a preset that has the same ID (but different key)
    # This should fail when add() is called
    json_with_existing_id = """{
  "new_key": {
    "id": "existing_preset",
    "mu": 0.15,
    "sigma": 0.25,
    "rho": 0.35
  }
}"""

    # This should fail with key mismatch first
    with pytest.raises(
        ValueError, match="Preset ID 'existing_preset' does not match its key 'new_key'"
    ):
        lib.load_json_str(json_with_existing_id)

    # Test case where keys match but we're adding to an existing library
    # (This demonstrates that add() would catch duplicates if keys matched)
    json_with_matching_key = """{
  "existing_preset": {
    "id": "existing_preset",
    "mu": 0.15,
    "sigma": 0.25,
    "rho": 0.35
  }
}"""

    # This should succeed because load_json_str clears the library first
    lib.load_json_str(json_with_matching_key)
    assert lib.get("existing_preset").mu == 0.15


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
