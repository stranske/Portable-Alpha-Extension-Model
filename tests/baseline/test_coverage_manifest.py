"""Distinguish catalog membership, observed config access and numerical evidence."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pandas as pd
import pytest
from baseline_kit import CoverageManifest, evaluate_direction, load_catalog

from . import adapter
from .conftest import CATALOG_PATH, REPO_ROOT
from .test_directional import _variant_patch

REPORT_PATH = REPO_ROOT / "docs" / "reports" / "baseline-coverage.md"


def _touched_inputs(catalog) -> set[str]:
    keys: set[str] = set()
    for scen in catalog.get("scenarios", []):
        keys.update((scen.get("vary_mult") or {}).keys())
        keys.update((scen.get("vary_set") or {}).keys())
    return keys


@functools.lru_cache(maxsize=1)
def _observations():
    catalog = load_catalog(CATALOG_PATH)
    reads: set[str] = set()
    baseline = adapter.run(read_keys=reads)
    proven: set[str] = set()
    for scen in catalog["scenarios"]:
        variant = adapter.run(_variant_patch(scen), read_keys=reads)
        if scen.get("enforce") and evaluate_direction(
            scen["direction"],
            adapter.metric(variant, scen["agent"], scen["metric"]),
            adapter.metric(baseline, scen["agent"], scen["metric"]),
        ):
            proven.update(_touched_inputs({"scenarios": [scen]}))
    return reads, proven


def _manifest() -> CoverageManifest:
    catalog = load_catalog(CATALOG_PATH)
    reads, _ = _observations()
    return CoverageManifest(
        all_keys=set(type(adapter._base_config()).model_fields),
        touched_keys=_touched_inputs(catalog),
        read_keys=set(reads),
        priority_params=list(catalog["priority_params"]),
        title="PAEM baseline coverage manifest",
    )


def _require_reads(manifest, required):
    missing = set(required) - manifest.read_keys
    assert not missing, "No computational read observed: " + ", ".join(sorted(missing))


def test_priority_inputs_are_exercised():
    manifest = _manifest()
    assert not manifest.priority_gaps
    assert not manifest.unknown_catalog_keys


def test_enforced_priority_scenarios_have_computational_reads():
    catalog = load_catalog(CATALOG_PATH)
    required = _touched_inputs(
        {"scenarios": [scen for scen in catalog["scenarios"] if scen.get("enforce")]}
    )
    _require_reads(_manifest(), required)


def test_serialization_only_bypass_does_not_satisfy_read_gate(monkeypatch):
    import pa_core.facade as facade

    def bypass(config, index, options):
        # The full validated config is serialized, but nothing computes with it.
        assert "internal_pa_financing_mean_month" in config.model_dump()
        return SimpleNamespace(summary=pd.DataFrame({"Agent": ["InternalPA"]}))

    monkeypatch.setattr(facade, "run_single", bypass)
    reads: set[str] = set()
    adapter.run({"internal_pa_financing_mean_month": 0.004}, read_keys=reads)
    assert reads == set()
    with pytest.raises(AssertionError, match="internal_pa_financing_mean_month"):
        _require_reads(SimpleNamespace(read_keys=reads), {"internal_pa_financing_mean_month"})


def test_recording_preserves_real_numerical_results():
    reads: set[str] = set()
    observed = adapter.run(read_keys=reads)
    pd.testing.assert_frame_equal(observed, adapter.run())
    assert {"mu_H", "sigma_H", "internal_pa_financing_mean_month"} <= reads


def test_emit_coverage_report():
    manifest = _manifest()
    _, proven = _observations()
    lines = [
        manifest.to_markdown().rstrip(),
        "",
        "## Evidence boundary",
        "",
        "Reads are recorded only around the real run_single call after validation.",
        "Bound model_dump serialization and later report traversal do not add reads.",
        "A read is access evidence, not numerical wiring proof. Enforced scenarios",
        "below passed directional comparisons against the same-seed funded control.",
        "Correlation scenarios vary six fields together: their proof is joint,",
        "not independent influence for each individual correlation.",
        "",
        "| Input | Catalog touched | Computational read | Behavioral evidence |",
        "| --- | --- | --- | --- |",
    ]
    for key in sorted(manifest.touched_keys):
        behavior = "enforced directional scenario" if key in proven else "unproven / report-only"
        lines.append(
            f"| `{key}` | yes | {'yes' if key in manifest.read_keys else 'not observed'} | {behavior} |"
        )
    lines += [
        "",
        "`internal_pa_financing_mean_month` funds InternalPA; the separate",
        "`internal_financing_mean_month` is the margin-financing input and is not",
        "substituted into that sleeve's directional claim.",
        "`active_share` remains report-only in this returns-mode scenario. Its",
        "validation/derived-agent construction is outside this runtime attribute",
        "recorder; absence of a read is not proof that the input is globally ignored.",
        "This bounded baseline is coordinated with broad coverage issue #2336;",
        "it makes no repository-wide coverage or universal parameter-influence claim.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    assert manifest.read_keys
