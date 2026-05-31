#!/usr/bin/env python3
"""Validate llm dependency pins."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

LLM_REQUIREMENTS = Path("tools/requirements-llm.txt")
LOCKFILE = Path("requirements.lock")
WORKFLOW_DRIFT_MARKER = "intentionally drift from pyproject.toml/requirements.lock"

LOCK_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s;#]+)" r"(?:\s*;\s*[^#]+)?(?:\s*#.*)?$"
)


def _read_llm_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(Requirement(line))
    return requirements


def _allows_workflow_drift(path: Path) -> bool:
    """Return true when requirements are documented as workflow-only pins."""
    return WORKFLOW_DRIFT_MARKER in path.read_text(encoding="utf-8")


def _read_lockfile_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = LOCK_LINE_RE.match(line)
        if not match:
            continue
        name = canonicalize_name(match.group("name"))
        pins[name] = match.group("version")
    return pins


def _validate_strict_workflow_pins(requirements: list[Requirement]) -> list[str]:
    errors: list[str] = []
    for requirement in requirements:
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            errors.append(f"- workflow pin must use one exact == version: {requirement.name}")
            continue
        try:
            Version(specifiers[0].version)
        except InvalidVersion:
            errors.append(
                f"- invalid workflow pin for {requirement.name}: " f"{specifiers[0].version!r}"
            )
    return errors


def main() -> int:
    if not LLM_REQUIREMENTS.exists():
        print(f"error: missing file {LLM_REQUIREMENTS}", file=sys.stderr)
        return 1
    if not LOCKFILE.exists():
        print(f"error: missing file {LOCKFILE}", file=sys.stderr)
        return 1

    llm_requirements = _read_llm_requirements(LLM_REQUIREMENTS)
    if _allows_workflow_drift(LLM_REQUIREMENTS):
        errors = _validate_strict_workflow_pins(llm_requirements)
        if errors:
            print("Workflow LLM requirement validation failed:")
            print("\n".join(errors))
            return 1
        print(
            "Workflow LLM requirement validation passed for "
            f"{len(llm_requirements)} standalone pins in {LLM_REQUIREMENTS}."
        )
        return 0

    lockfile_pins = _read_lockfile_pins(LOCKFILE)

    errors: list[str] = []
    for requirement in llm_requirements:
        package_name = canonicalize_name(requirement.name)
        if package_name not in lockfile_pins:
            errors.append(f"- missing in lockfile: {requirement.name}")
            continue

        locked_version_text = lockfile_pins[package_name]
        try:
            locked_version = Version(locked_version_text)
        except InvalidVersion:
            errors.append(
                f"- invalid locked version for {requirement.name}: {locked_version_text!r}"
            )
            continue

        if requirement.specifier and not requirement.specifier.contains(
            locked_version, prereleases=True
        ):
            errors.append(
                f"- incompatible pin for {requirement.name}: "
                f"lockfile has {locked_version}, requirement is {requirement.specifier}"
            )

    if errors:
        print("Lockfile validation failed:")
        print("\n".join(errors))
        return 1

    print(
        "Lockfile validation passed for "
        f"{len(llm_requirements)} llm dependencies in {LLM_REQUIREMENTS}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
