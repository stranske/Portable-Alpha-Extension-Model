"""Coverage manifest -- which priority inputs are exercised; emit the report."""

from __future__ import annotations

from pathlib import Path

from baseline_kit import CoverageManifest, load_catalog

from .conftest import CATALOG_PATH, REPO_ROOT

REPORT_PATH = REPO_ROOT / "docs" / "reports" / "baseline-coverage.md"


def _touched_inputs(catalog) -> set[str]:
    keys: set[str] = set()
    for scen in catalog.get("scenarios", []):
        keys.update((scen.get("vary_mult") or {}).keys())
        keys.update((scen.get("vary_set") or {}).keys())
    return keys


def _manifest() -> CoverageManifest:
    catalog = load_catalog(CATALOG_PATH)
    priority = list(catalog.get("priority_params", []))
    touched = _touched_inputs(catalog)
    return CoverageManifest(
        all_keys=set(priority) | touched,
        touched_keys=touched,
        priority_params=priority,
        title="PAEM baseline coverage manifest",
    )


def test_priority_inputs_are_exercised():
    m = _manifest()
    assert not m.priority_gaps, "Priority inputs with no scenario: " + ", ".join(m.priority_gaps)


def test_emit_coverage_report():
    m = _manifest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(m.to_markdown())
    assert REPORT_PATH.exists()
