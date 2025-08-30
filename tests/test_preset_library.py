from pathlib import Path
import sys

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
