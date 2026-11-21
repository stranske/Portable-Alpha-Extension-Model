"""Tier 1 directional sensibility: each scenario's input change moves a real
simulation metric in the economically expected direction (variant vs baseline)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tomllib

import pytest
from baseline_kit import evaluate_direction, load_catalog
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from . import adapter
from .conftest import CATALOG_PATH, REPO_ROOT

_SCENARIOS = load_catalog(CATALOG_PATH)["scenarios"]
_IDS = [s["id"] for s in _SCENARIOS]


@pytest.mark.skipif(shutil.which("make") is None, reason="requires Make")
def test_install_dev_selects_baseline_kit_extra():
    """The developer setup must request an extra that provides the baseline harness."""
    recipe = subprocess.run(
        ["make", "--dry-run", "install-dev"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    args = shlex.split(recipe)
    editable = args[args.index("-e") + 1]

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = Requirement(editable.replace(".", project["name"], 1)).extras
    dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for extra in extras
        for requirement in project["optional-dependencies"][extra]
    }
    assert "app-baseline-kit" in dependencies, f"make install-dev uses {editable!r}"


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


def test_internal_financing_contribution_reaches_internal_pa(baseline_summary):
    """Real funded consumer must respond even though a config read alone could pass."""
    field = "internal_pa_financing_mean_month"
    reads: set[str] = set()
    variant = adapter.run({field: 0.004}, read_keys=reads)
    assert field in reads
    control = adapter.metric(baseline_summary, "InternalPA", "terminal_AnnReturn")
    financed = adapter.metric(variant, "InternalPA", "terminal_AnnReturn")
    assert financed < control - 0.001, (control, financed)
