"""Preset library for alpha streams.

Provides CRUD operations and import/export to YAML or JSON for presets
that define expected return (mu), volatility (sigma), and correlation
with the index (rho).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable

import json
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
        self._presets: Dict[str, AlphaPreset] = {}
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
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {pid: asdict(p) for pid, p in self._presets.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "PresetLibrary":
        presets = [AlphaPreset(**v) for v in data.values()]
        return cls(presets)

    # YAML/JSON import/export
    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict()))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PresetLibrary":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> "PresetLibrary":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # Convenience methods for strings (used by dashboard)
    def to_yaml_str(self) -> str:
        return yaml.safe_dump(self.to_dict())

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict())

    def load_yaml_str(self, text: str) -> None:
        data = yaml.safe_load(text) or {}
        # Validate for duplicate IDs before clearing existing presets
        ids = [p.get("id") for p in data.values()]
        duplicate_ids = {id for id in ids if ids.count(id) > 1}
        if duplicate_ids:
            raise ValueError(f"Duplicate preset IDs found in input: {', '.join(duplicate_ids)}")
        self._presets = {}
        for p in data.values():
            self.add(AlphaPreset(**p))
    def load_json_str(self, text: str) -> None:
        data = json.loads(text)
        self._presets = {}
        for p in data.values():
            self.add(AlphaPreset(**p))

    @property
    def presets(self) -> Dict[str, AlphaPreset]:
        return self._presets


__all__ = ["AlphaPreset", "PresetLibrary"]