from __future__ import annotations

import hashlib
import json
import subprocess
import warnings as _warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

#: Emitted (and recorded in the manifest) when a run has no explicit seed.
SEED_REPRODUCIBILITY_WARNING = (
    "Run executed without an explicit --seed (seed=None): results use a "
    "non-deterministic RNG and cannot be reproduced. Pass --seed to make the "
    "run reproducible."
)


@dataclass
class Manifest:
    git_commit: str
    timestamp: str
    seed: int | None
    substream_ids: Mapping[str, str] | None
    config: Mapping[str, Any]
    data_files: Mapping[str, str]
    cli_args: Mapping[str, Any]
    config_hash: str | None = None
    backend: str | None = None
    run_log: str | None = None
    previous_run: str | None = None
    run_timing: Mapping[str, Any] | None = None
    warnings: Sequence[Mapping[str, Any]] | None = None
    cost: Mapping[str, Any] | None = None
    data_quality: Mapping[str, Any] | None = None


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

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def write(
        self,
        *,
        config_path: str | Path,
        config_snapshot: str | None = None,
        config_snapshot_bytes: bytes | None = None,
        data_files: Sequence[str | Path],
        seed: int | None,
        substream_ids: Mapping[str, str] | None = None,
        cli_args: Mapping[str, Any],
        backend: str | None = None,
        run_log: str | Path | None = None,
        previous_run: str | Path | None = None,
        run_timing: Mapping[str, Any] | None = None,
        warnings: Sequence[Mapping[str, Any]] | None = None,
        cost: Mapping[str, Any] | None = None,
        data_quality: Mapping[str, Any] | None = None,
    ) -> None:
        """Write manifest to ``self.path``.

        ``warnings``, ``cost``, and ``data_quality`` are additive optional
        fields (see ``MANIFEST_OPTIONAL_FIELDS``); they default to ``None`` so
        existing callers are unaffected and may be finalized post-run by the CLI.
        """

        repo_root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            commit = "unknown"
        if config_snapshot is not None:
            cfg_text = config_snapshot
            cfg_bytes = (
                config_snapshot_bytes
                if config_snapshot_bytes is not None
                else config_snapshot.encode()
            )
        else:
            cfg_bytes = Path(config_path).read_bytes()
            cfg_text = cfg_bytes.decode()
        cfg = yaml.safe_load(cfg_text)
        config_hash = self._hash_bytes(cfg_bytes)
        hashes = {str(Path(p)): self._hash_file(p) for p in data_files if Path(p).exists()}
        timing = dict(run_timing) if run_timing else None
        if seed is None:
            with _warnings.catch_warnings():
                _warnings.simplefilter("always", UserWarning)
                _warnings.warn(SEED_REPRODUCIBILITY_WARNING, UserWarning, stacklevel=2)
        manifest = Manifest(
            git_commit=commit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=seed,
            substream_ids=dict(substream_ids) if substream_ids is not None else None,
            config=cfg,
            data_files=hashes,
            cli_args=dict(cli_args),
            config_hash=config_hash,
            backend=backend,
            run_log=str(run_log) if run_log else None,
            previous_run=str(previous_run) if previous_run else None,
            run_timing=timing,
            warnings=[dict(w) for w in warnings] if warnings is not None else None,
            cost=dict(cost) if cost is not None else None,
            data_quality=dict(data_quality) if data_quality is not None else None,
        )
        self.path.write_text(json.dumps(asdict(manifest), indent=2))
