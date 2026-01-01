from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

REGISTRY_DIRNAME = ".pa_registry"
SCENARIO_ID_LEN = 12


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_hash: str
    created_at: str
    config: Mapping[str, Any]
    index_path: str
    index_hash: str
    seed: int | None
    code_version: str


@dataclass(frozen=True)
class ScenarioSummary:
    scenario_id: str
    created_at: str
    index_hash: str
    seed: int | None
    code_version: str


def _registry_dir() -> Path:
    return Path(REGISTRY_DIRNAME)


def _ensure_registry_dir() -> Path:
    path = _registry_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_config(config: Any) -> Mapping[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a mapping or pydantic model")


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_index_series(index_series: "pd.Series") -> str:
    import pandas as pd

    hasher = hashlib.sha256()
    hashed = pd.util.hash_pandas_object(index_series, index=True).to_numpy()
    hasher.update(hashed.tobytes())
    return hasher.hexdigest()


def _prepare_index_series(index: Any) -> tuple["pd.Series", str]:
    import pandas as pd

    if isinstance(index, (str, Path)):
        from .data.loaders import load_index_returns
        from .units import get_index_series_unit, normalize_index_series

        index_path = str(index)
        series = load_index_returns(index)
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        if not isinstance(series, pd.Series):
            raise ValueError("Index data must be a pandas Series")
        series = normalize_index_series(series, get_index_series_unit())
        series.attrs.setdefault("source_path", index_path)
        return series, index_path

    if isinstance(index, pd.Series):
        source_path = index.attrs.get("source_path")
        return index, str(source_path) if source_path else "<in-memory>"

    raise TypeError("index must be a path or pandas Series")


def _get_code_version() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _build_scenario_hash(
    *,
    config: Any,
    index: Any,
    seed: int | None,
) -> tuple[str, str, str, Mapping[str, Any], "pd.Series", str]:
    config_data = _normalize_config(config)
    series, index_path = _prepare_index_series(index)
    index_hash = _hash_index_series(series)
    code_version = _get_code_version()
    payload = {
        "config": config_data,
        "index_hash": index_hash,
        "code_version": code_version,
        "seed": seed,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest, index_hash, code_version, config_data, series, index_path


def compute_scenario_id(config: Any, index_path: str | Path, seed: int | None) -> str:
    """Return short scenario id: sha256(config + index_hash + code_version + seed)."""
    digest, _, _, _, _, _ = _build_scenario_hash(
        config=config,
        index=index_path,
        seed=seed,
    )
    return digest[:SCENARIO_ID_LEN]


def register(config: Any, index: Any, seed: int | None) -> str:
    """Register scenario and return its short scenario id."""
    (
        digest,
        index_hash,
        code_version,
        config_data,
        _,
        index_path,
    ) = _build_scenario_hash(config=config, index=index, seed=seed)
    scenario_id = digest[:SCENARIO_ID_LEN]
    registry_dir = _ensure_registry_dir()
    scenario_path = registry_dir / f"{scenario_id}.json"
    scenario = Scenario(
        scenario_id=scenario_id,
        scenario_hash=digest,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config_data,
        index_path=index_path,
        index_hash=index_hash,
        seed=seed,
        code_version=code_version,
    )
    if scenario_path.exists():
        existing = json.loads(scenario_path.read_text())
        if existing.get("scenario_hash") != digest:
            raise ValueError("Scenario ID collision detected")
        return scenario_id
    scenario_path.write_text(_canonical_json(scenario.__dict__))
    return scenario_id


def get(scenario_id: str) -> Scenario:
    """Return Scenario metadata for the given id."""
    scenario_path = _registry_dir() / f"{scenario_id}.json"
    if not scenario_path.exists():
        raise KeyError(f"Scenario not found: {scenario_id}")
    data = json.loads(scenario_path.read_text())
    return Scenario(**data)


def list() -> list[ScenarioSummary]:
    """Return summaries for registered scenarios."""
    registry_dir = _registry_dir()
    if not registry_dir.exists():
        return []
    summaries: list[ScenarioSummary] = []
    for path in sorted(registry_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summaries.append(
            ScenarioSummary(
                scenario_id=data.get("scenario_id", path.stem),
                created_at=data.get("created_at", ""),
                index_hash=data.get("index_hash", ""),
                seed=data.get("seed"),
                code_version=data.get("code_version", ""),
            )
        )
    return summaries
