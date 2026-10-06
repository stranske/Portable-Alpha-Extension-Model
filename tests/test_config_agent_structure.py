"""Public config loading must distinguish absent metadata from malformed containers."""

from __future__ import annotations

from collections import UserDict
from types import MappingProxyType

import pytest

from pa_core.config import AgentConfig, load_config


def _config(agents: object) -> dict[str, object]:
    return {
        "N_SIMULATIONS": 10,
        "N_MONTHS": 12,
        "financing_mode": "broadcast",
        "analysis_mode": "returns",
        "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        "agents": agents,
    }


def _benchmark(**updates: object) -> dict[str, object]:
    return {
        "name": "Base",
        "capital": 1000.0,
        "beta_share": 100.0,
        "alpha_share": 0.0,
        **updates,
    }


@pytest.mark.parametrize("extra", [[], (), "", False, 0, ["metadata"], "metadata"])
def test_agent_extra_rejects_non_mapping_even_when_empty(extra: object) -> None:
    with pytest.raises(ValueError, match=r"agents\[0\]\.extra must be a mapping"):
        load_config(_config([_benchmark(extra=extra)]))


@pytest.mark.parametrize("agent", [None, 42, "Base"])
def test_agent_entry_rejects_non_mapping(agent: object) -> None:
    with pytest.raises(ValueError, match=r"agents\[0\] must be a mapping"):
        load_config(_config([agent]))


@pytest.mark.parametrize("missing", ["name", "capital", "beta_share", "alpha_share"])
def test_agent_entry_reports_missing_required_key(missing: str) -> None:
    agent = _benchmark()
    del agent[missing]
    with pytest.raises(ValueError, match=rf"agents\[0\] missing keys: .*{missing}"):
        load_config(_config([agent]))


@pytest.mark.parametrize("extra", [None, {}, {"desk": "Equities", "tag": 0}])
def test_agent_extra_accepts_none_or_mapping_and_preserves_values(extra: object) -> None:
    cfg = load_config(_config([_benchmark(extra=extra)]))
    assert cfg.agents[0].extra == ({} if extra is None else extra)
    assert cfg.agents[0].capital == 1000.0
    assert cfg.agents[0].beta_share == 1.0
    assert cfg.agents[0].alpha_share == 0.0


@pytest.mark.parametrize("mapping_type", [UserDict, MappingProxyType])
@pytest.mark.parametrize("metadata", [{}, {"desk": "Equities", "tag": 0}])
def test_agent_extra_accepts_mapping_implementations(mapping_type, metadata) -> None:
    extra = mapping_type(metadata)
    cfg = load_config(_config([_benchmark(extra=extra)]))
    assert isinstance(cfg.agents[0].extra, dict)
    assert cfg.agents[0].extra == metadata
    assert cfg.agents[0].beta_share == 1.0
    assert dict(extra) == metadata


def test_agent_extra_can_be_omitted() -> None:
    cfg = load_config(_config([_benchmark()]))
    assert cfg.agents[0].extra == {}
    assert cfg.agents[0].beta_share == 1.0


def test_agent_model_instance_preserves_metadata_and_normalizes_shares() -> None:
    agent = AgentConfig(**_benchmark(extra={"desk": "Equities"}))
    cfg = load_config(_config([agent]))
    assert cfg.agents[0].name == "Base"
    assert cfg.agents[0].capital == 1000.0
    assert cfg.agents[0].beta_share == 1.0
    assert cfg.agents[0].alpha_share == 0.0
    assert cfg.agents[0].extra == {"desk": "Equities"}
