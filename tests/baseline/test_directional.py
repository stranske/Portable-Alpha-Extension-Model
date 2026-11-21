"""Tier 1 directional sensibility: each scenario's input change moves a real
simulation metric in the economically expected direction (variant vs baseline)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from baseline_kit import evaluate_direction, load_catalog
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from . import adapter
from .conftest import CATALOG_PATH, REPO_ROOT

_SCENARIOS = load_catalog(CATALOG_PATH)["scenarios"]
_IDS = [s["id"] for s in _SCENARIOS]


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("make") is None, reason="requires POSIX Make"
)
def test_install_dev_selects_baseline_kit_extra(tmp_path: Path):
    """The developer setup must request an extra that provides the baseline harness."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for filename in ("Makefile", "pyproject.toml"):
        shutil.copy2(REPO_ROOT / filename, checkout / filename)

    # Record the actual install arguments without changing the test environment.
    pip = tmp_path / "pip"
    pip.write_text(
        "#!/bin/sh\n"
        'if [ "${0##*/}" = python ]; then\n'
        '    [ "$1" = -m ] && [ "$2" = pip ] || exit 2\n'
        "    shift 2\n"
        "fi\n"
        '[ "$1" = install ] || exit 2\n'
        'printf "%s\\n" "$@" ""\n',
        encoding="utf-8",
    )
    pip.chmod(0o755)
    (tmp_path / "python").symlink_to(pip)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{os.environ.get('PATH', os.defpath)}"
    for name in ("GNUMAKEFLAGS", "MAKEFLAGS", "MAKEFILES", "MFLAGS"):
        env.pop(name, None)
    result = subprocess.run(
        ["make", "--silent", "--no-print-directory", "install-dev"],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    editable = []
    for invocation in result.stdout.strip().split("\n\n"):
        args = invocation.splitlines()
        if "-e" in args:
            editable.append(args[args.index("-e") + 1])

    with (checkout / "pyproject.toml").open("rb") as metadata:
        project = tomllib.load(metadata)["project"]
    dependencies = {
        canonicalize_name(Requirement(requirement).name)
        for target in editable
        if target == "." or target.startswith(".[")
        for extra in Requirement(project["name"] + target[1:]).extras
        for requirement in project["optional-dependencies"][extra]
    }
    assert "app-baseline-kit" in dependencies, f"make install-dev requested {editable!r}"


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
