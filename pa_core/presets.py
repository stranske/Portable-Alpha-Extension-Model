"""Preset library for alpha streams.

Provides CRUD operations and import/export to YAML or JSON for presets
that define expected return (mu), volatility (sigma), and correlation
with the index (rho).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class AlphaPreset:
    """Parameters describing an alpha stream."""

    id: str
    mu: float
    sigma: float
    rho: float


class PresetLibrary:
    """Manage a collection of :class:`AlphaPreset` objects."""

    def __init__(self, presets: Iterable[AlphaPreset] | None = None) -> None:
        self._presets: dict[str, AlphaPreset] = {}
        if presets:
            for p in presets:
                self.add(p)

    # CRUD operations
    def add(self, preset: AlphaPreset) -> None:
        """Add ``preset`` to the library.

        Raises
        ------
        ValueError
            If a preset with the same id already exists.
        """
        if preset.id in self._presets:
            raise ValueError(f"preset '{preset.id}' already exists")
        self._presets[preset.id] = preset

    def get(self, preset_id: str) -> AlphaPreset:
        return self._presets[preset_id]

    def update(self, preset: AlphaPreset) -> None:
        if preset.id not in self._presets:
            raise KeyError(preset.id)
        self._presets[preset.id] = preset

    def delete(self, preset_id: str) -> None:
        """Delete the preset with the given id.

        Raises
        ------
        KeyError
            If the preset with the given id does not exist.
        """
        if preset_id not in self._presets:
            raise KeyError(preset_id)
        self._presets.pop(preset_id)

    # Serialization helpers
    def to_dict(self) -> dict[str, dict[str, float | str]]:
        # Ensure the nested dict contains the id key as well for round-trip
        out: dict[str, dict[str, float | str]] = {}
        for pid, p in self._presets.items():
            d = asdict(p)
            d["id"] = p.id
            out[pid] = d
        return out

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float | str]]) -> PresetLibrary:
        presets = [
            AlphaPreset(
                id=k,
                mu=float(v["mu"]),
                sigma=float(v["sigma"]),
                rho=float(v["rho"]),
            )
            for k, v in data.items()
        ]
        return cls(presets)

    # YAML/JSON import/export
    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict()))

    @classmethod
    def from_yaml(cls, path: str | Path) -> PresetLibrary:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> PresetLibrary:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # Convenience methods for strings (used by dashboard)
    def to_yaml_str(self) -> str:
        return str(yaml.safe_dump(self.to_dict()))

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict())

    def load_yaml_str(self, text: str) -> None:
        data = yaml.safe_load(text) or {}
        # Check for duplicate IDs in the input data
        ids = [p.get("id") for p in data.values()]
        duplicate_ids = {id for id in ids if ids.count(id) > 1}
        if duplicate_ids:
            raise ValueError(f"Duplicate preset IDs found in input: {', '.join(duplicate_ids)}")

        # Validate that each preset.id matches its dictionary key
        for key, preset_data in data.items():
            preset_id = preset_data.get("id")
            if preset_id != key:
                raise ValueError(f"Preset ID '{preset_id}' does not match its key '{key}'")

        self._presets = {}
        for p in data.values():
            preset = AlphaPreset(**p)
            self._presets[preset.id] = preset

    def load_json_str(self, text: str) -> None:
        data = json.loads(text)
        # Validate that each preset.id matches its dictionary key
        for key, preset_data in data.items():
            preset_id = preset_data.get("id")
            if preset_id != key:
                raise ValueError(f"Preset ID '{preset_id}' does not match its key '{key}'")

        self._presets = {}
        for p in data.values():
            preset = AlphaPreset(**p)
            self._presets[preset.id] = preset

    @property
    def presets(self) -> dict[str, AlphaPreset]:
        return self._presets


__all__ = ["AlphaPreset", "PresetLibrary"]
