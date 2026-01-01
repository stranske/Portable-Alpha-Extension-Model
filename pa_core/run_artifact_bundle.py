from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import hashlib
import json
import shutil


@dataclass(slots=True)
class RunArtifact:
    """Portable snapshot describing a single run's reproducible outputs."""

    config: str
    index_hash: str
    seed: int | None
    manifest: Mapping[str, Any] | None
    outputs: Mapping[str, str]


class RunArtifactBundle:
    """Bundle that stores run artifacts and verifies integrity on disk."""

    _META_FILE = "bundle.json"
    _CONFIG_FILE = "config.yaml"
    _MANIFEST_FILE = "manifest.json"
    _OUTPUTS_DIR = "outputs"

    def __init__(self, artifact: RunArtifact, root: Path | None = None) -> None:
        self.artifact = artifact
        self.root = root

    def save(self, path: str | Path) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        outputs_dir = root / self._OUTPUTS_DIR
        outputs_dir.mkdir(parents=True, exist_ok=True)

        config_path = root / self._CONFIG_FILE
        config_path.write_text(self.artifact.config)

        manifest_path: Path | None = None
        if self.artifact.manifest is not None:
            manifest_path = root / self._MANIFEST_FILE
            manifest_path.write_text(json.dumps(self.artifact.manifest, indent=2))

        outputs_map: dict[str, str] = {}
        output_hashes: dict[str, str] = {}
        for name, src in self.artifact.outputs.items():
            src_path = Path(src)
            dest_rel = self._sanitize_output_path(name)
            dest_path = outputs_dir / dest_rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            outputs_map[name] = str(dest_path.relative_to(root))
            output_hashes[name] = self._hash_file(dest_path)

        bundle_meta = {
            "config_path": str(config_path.relative_to(root)),
            "index_hash": self.artifact.index_hash,
            "seed": self.artifact.seed,
            "manifest_path": (
                str(manifest_path.relative_to(root)) if manifest_path else None
            ),
            "outputs": outputs_map,
            "hashes": {
                "config": self._hash_file(config_path),
                "manifest": self._hash_file(manifest_path) if manifest_path else None,
                "outputs": output_hashes,
            },
        }
        meta_path = root / self._META_FILE
        meta_path.write_text(json.dumps(bundle_meta, indent=2))
        self.root = root
        return root

    @classmethod
    def load(cls, path: str | Path) -> "RunArtifactBundle":
        root = Path(path)
        meta_path = root / cls._META_FILE
        bundle_meta = json.loads(meta_path.read_text())

        config_text = (root / bundle_meta["config_path"]).read_text()
        manifest_data = None
        manifest_path = bundle_meta.get("manifest_path")
        if manifest_path:
            manifest_data = json.loads((root / manifest_path).read_text())

        outputs = {
            name: str(root / relpath) for name, relpath in bundle_meta["outputs"].items()
        }

        artifact = RunArtifact(
            config=config_text,
            index_hash=bundle_meta["index_hash"],
            seed=bundle_meta.get("seed"),
            manifest=manifest_data,
            outputs=outputs,
        )
        return cls(artifact, root=root)

    def verify(self) -> bool:
        if self.root is None:
            raise ValueError("Bundle root is not set; load or save the bundle first.")

        meta_path = self.root / self._META_FILE
        if not meta_path.exists():
            return False

        bundle_meta = json.loads(meta_path.read_text())
        config_path = self.root / bundle_meta["config_path"]
        if not config_path.exists():
            return False
        if self._hash_file(config_path) != bundle_meta["hashes"]["config"]:
            return False

        manifest_path = bundle_meta.get("manifest_path")
        if manifest_path:
            manifest_file = self.root / manifest_path
            if not manifest_file.exists():
                return False
            expected_manifest_hash = bundle_meta["hashes"].get("manifest")
            if expected_manifest_hash and self._hash_file(manifest_file) != expected_manifest_hash:
                return False

        outputs_meta: Mapping[str, str] = bundle_meta.get("outputs", {})
        output_hashes: Mapping[str, str] = bundle_meta.get("hashes", {}).get("outputs", {})
        for name, relpath in outputs_meta.items():
            output_path = self.root / relpath
            if not output_path.exists():
                return False
            expected_hash = output_hashes.get(name)
            if expected_hash and self._hash_file(output_path) != expected_hash:
                return False

        return True

    @staticmethod
    def _hash_file(path: Path | None) -> str:
        if path is None:
            return ""
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            hasher.update(handle.read())
        return hasher.hexdigest()

    @staticmethod
    def _sanitize_output_path(name: str) -> Path:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            return Path(path.name)
        return path
