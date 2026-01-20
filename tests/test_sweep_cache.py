from collections import OrderedDict

import pandas as pd
import pytest

from pa_core import sweep as sweep_module
from pa_core.config import load_config
from pa_core.sweep import clear_sweep_cache, run_parameter_sweep_cached, sweep_results_to_dataframe


def test_cached_sweep_deterministic():
    clear_sweep_cache()
    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    res1 = run_parameter_sweep_cached(cfg, idx, seed=123)
    res2 = run_parameter_sweep_cached(cfg, idx, seed=123)
    # cache should return same object
    assert res1 is res2

    df = sweep_results_to_dataframe(res1)
    # expect multiple scenarios and standard metrics
    assert df.shape[0] > 1
    for col in ["terminal_AnnReturn", "monthly_AnnVol", "monthly_TE"]:
        assert col in df.columns


def test_cached_sweep_progress_callback_invoked():
    clear_sweep_cache()
    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    run_parameter_sweep_cached(cfg, idx, seed=321)

    calls: list[tuple[int, int]] = []

    def _progress(current: int, total: int) -> None:
        calls.append((current, total))

    results = run_parameter_sweep_cached(cfg, idx, seed=321, progress=_progress)
    assert calls == [(len(results), len(results))]


def test_sweep_cache_evicts_least_recently_used(monkeypatch):
    clear_sweep_cache()
    monkeypatch.setattr(sweep_module, "SWEEP_CACHE_MAX_ENTRIES", 2)

    def _fake_run_parameter_sweep(*_args, **_kwargs):
        return [{"combination_id": 0, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr(sweep_module, "run_parameter_sweep", _fake_run_parameter_sweep)

    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    res1 = run_parameter_sweep_cached(cfg, idx, seed=1)
    res2 = run_parameter_sweep_cached(cfg, idx, seed=2)
    res1_again = run_parameter_sweep_cached(cfg, idx, seed=1)
    assert res1_again is res1
    assert len(sweep_module._SWEEP_CACHE) == 2

    run_parameter_sweep_cached(cfg, idx, seed=3)
    assert len(sweep_module._SWEEP_CACHE) == 2
    res2_again = run_parameter_sweep_cached(cfg, idx, seed=2)
    assert res2_again is not res2


def test_sweep_cache_respects_max_entries(monkeypatch):
    clear_sweep_cache()
    monkeypatch.setattr(sweep_module, "SWEEP_CACHE_MAX_ENTRIES", 1)

    def _fake_run_parameter_sweep(*_args, **_kwargs):
        return [{"combination_id": 0, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr(sweep_module, "run_parameter_sweep", _fake_run_parameter_sweep)

    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    run_parameter_sweep_cached(cfg, idx, seed=10)
    run_parameter_sweep_cached(cfg, idx, seed=11)
    run_parameter_sweep_cached(cfg, idx, seed=12)
    assert len(sweep_module._SWEEP_CACHE) == 1
    assert isinstance(sweep_module._SWEEP_CACHE, OrderedDict)


def test_sweep_cache_trims_when_max_reduced(monkeypatch):
    clear_sweep_cache()
    monkeypatch.setattr(sweep_module, "SWEEP_CACHE_MAX_ENTRIES", 3)

    def _fake_run_parameter_sweep(*_args, **_kwargs):
        return [{"combination_id": 0, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr(sweep_module, "run_parameter_sweep", _fake_run_parameter_sweep)

    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    run_parameter_sweep_cached(cfg, idx, seed=1)
    res2 = run_parameter_sweep_cached(cfg, idx, seed=2)
    run_parameter_sweep_cached(cfg, idx, seed=3)
    assert len(sweep_module._SWEEP_CACHE) == 3

    monkeypatch.setattr(sweep_module, "SWEEP_CACHE_MAX_ENTRIES", 1)
    res2_again = run_parameter_sweep_cached(cfg, idx, seed=2)
    key2 = sweep_module._make_cache_key(cfg, idx, 2)
    assert res2_again is res2
    assert list(sweep_module._SWEEP_CACHE.keys()) == [key2]


@pytest.mark.parametrize("max_entries", [0, -1])
def test_sweep_cache_disabled_when_max_non_positive(monkeypatch, max_entries):
    clear_sweep_cache()
    monkeypatch.setattr(sweep_module, "SWEEP_CACHE_MAX_ENTRIES", max_entries)
    call_count = 0

    def _fake_run_parameter_sweep(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return [{"combination_id": call_count, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr(sweep_module, "run_parameter_sweep", _fake_run_parameter_sweep)

    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    res1 = run_parameter_sweep_cached(cfg, idx, seed=7)
    res2 = run_parameter_sweep_cached(cfg, idx, seed=7)

    assert call_count == 2
    assert res2 is not res1
    assert len(sweep_module._SWEEP_CACHE) == 0


def test_clear_sweep_cache_empties_cache(monkeypatch):
    clear_sweep_cache()

    def _fake_run_parameter_sweep(*_args, **_kwargs):
        return [{"combination_id": 0, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr(sweep_module, "run_parameter_sweep", _fake_run_parameter_sweep)

    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01, 0.0])
    res1 = run_parameter_sweep_cached(cfg, idx, seed=99)
    clear_sweep_cache()
    res2 = run_parameter_sweep_cached(cfg, idx, seed=99)
    assert res2 is not res1
