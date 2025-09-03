from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from .backend import get_backend


@dataclass
class Manifest:
    git_commit: str
    timestamp: str
    seed: int | None
    config: Mapping[str, Any]
    data_files: Mapping[str, str]
    cli_args: Mapping[str, Any]
    backend: str | None = None
    run_log: str | None = None
    previous_run: str | None = None


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
    backend: str | None = None,
    run_log: str | Path | None = None,
        previous_run: str | None = None,
    ) -> None:
        """Write manifest to ``self.path``."""

        repo_root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            commit = "unknown"
        cfg = yaml.safe_load(Path(config_path).read_text())
        hashes = {
            str(Path(p)): self._hash_file(p) for p in data_files if Path(p).exists()
        }
        manifest = Manifest(
            git_commit=commit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=seed,
            config=cfg,
            data_files=hashes,
            cli_args=dict(cli_args),
            backend=backend,
            run_log=str(run_log) if run_log else None,
            previous_run=previous_run,
        )
        self.path.write_text(json.dumps(asdict(manifest), indent=2))
