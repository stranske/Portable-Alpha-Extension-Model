"""Tier 1 directional sensibility: each scenario's input change moves a real
simulation metric in the economically expected direction (variant vs baseline)."""

from __future__ import annotations

import pytest
from baseline_kit import evaluate_direction, load_catalog

from . import adapter
from .conftest import CATALOG_PATH

_SCENARIOS = load_catalog(CATALOG_PATH)["scenarios"]
_IDS = [s["id"] for s in _SCENARIOS]


def _variant_patch(scen) -> dict:
    patch: dict = {}
    for field, mult in (scen.get("vary_mult") or {}).items():
        patch[field] = adapter.base_field(field) * float(mult)
    patch.update(scen.get("vary_set") or {})
    return patch


@pytest.mark.parametrize("scen", _SCENARIOS, ids=_IDS)
def test_directional(scen, baseline_summary, record_property):
    metric_col = scen["metric"]
    control = adapter.metric(baseline_summary, scen["agent"], metric_col)
    variant = adapter.metric(adapter.run(_variant_patch(scen)), scen["agent"], metric_col)
    holds = evaluate_direction(scen["direction"], variant, control)
    msg = (
        f"{scen['id']}: {scen['agent']}.{metric_col} "
        f"control={control:.6g} variant={variant:.6g} {scen['direction']} -> {holds}"
    )
    record_property("directional", msg)
    if scen.get("enforce"):
        assert holds, "Economically wrong direction -- " + msg
    elif not holds:
        pytest.skip("[report-only / finding] " + msg)
