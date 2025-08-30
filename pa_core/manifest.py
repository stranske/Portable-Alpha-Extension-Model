from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import hashlib
import yaml


@dataclass
class Manifest:
    git_commit: str
    timestamp: str
    seed: int | None
    config: Mapping[str, Any]
    data_files: Mapping[str, str]
    cli_args: Mapping[str, Any]


class ManifestWriter:
    """Write a reproducibility manifest for simulation runs."""

    def __init__(self, path: str | Path = "manifest.json") -> None:
        self.path = Path(path)

    @staticmethod
    def _hash_file(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    def write(
        self,
        *,
        config_path: str | Path,
        data_files: Sequence[str | Path],
        seed: int | None,
        cli_args: Mapping[str, Any],
    ) -> None:
        """Write manifest to ``self.path``."""

        repo_root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
        except Exception:
            commit = "unknown"
        cfg = yaml.safe_load(Path(config_path).read_text())
        hashes = {
            str(Path(p)): self._hash_file(p)
            for p in data_files
            if Path(p).exists()
        }
        manifest = Manifest(
            git_commit=commit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=seed,
            config=cfg,
            data_files=hashes,
            cli_args=dict(cli_args),
        )
        self.path.write_text(json.dumps(asdict(manifest), indent=2))
