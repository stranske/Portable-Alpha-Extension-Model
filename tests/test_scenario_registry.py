from __future__ import annotations

from pathlib import Path

import pytest

import pa_core.scenario_registry as registry
from pa_core.config import ModelConfig


def _write_index_csv(path: Path) -> None:
    path.write_text("Date,Monthly_TR\n2020-01-31,0.01\n2020-02-29,0.02\n2020-03-31,0.03\n")


def test_compute_scenario_id_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "_get_code_version", lambda: "test-version")
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    index_path = tmp_path / "index.csv"
    _write_index_csv(index_path)

    scenario_id = registry.compute_scenario_id(cfg, index_path, seed=123)
    scenario_id_again = registry.compute_scenario_id(cfg, index_path, seed=123)

    assert scenario_id == scenario_id_again

    different_seed = registry.compute_scenario_id(cfg, index_path, seed=124)
    assert scenario_id != different_seed

    different_cfg = cfg.model_copy(update={"N_MONTHS": 2})
    different_cfg_id = registry.compute_scenario_id(different_cfg, index_path, seed=123)
    assert scenario_id != different_cfg_id


def test_compute_scenario_id_includes_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    index_path = tmp_path / "index.csv"
    _write_index_csv(index_path)

    monkeypatch.setattr(registry, "_get_code_version", lambda: "version-a")
    scenario_id_a = registry.compute_scenario_id(cfg, index_path, seed=123)

    monkeypatch.setattr(registry, "_get_code_version", lambda: "version-b")
    scenario_id_b = registry.compute_scenario_id(cfg, index_path, seed=123)

    assert scenario_id_a != scenario_id_b


def test_register_get_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "_get_code_version", lambda: "test-version")
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    index_path = tmp_path / "index.csv"
    _write_index_csv(index_path)

    scenario_id = registry.register(cfg, index_path, seed=42)

    scenario = registry.get(scenario_id)
    assert scenario.scenario_id == scenario_id
    assert scenario.index_hash
    assert scenario.code_version == "test-version"

    summaries = registry.list()
    assert any(summary.scenario_id == scenario_id for summary in summaries)
