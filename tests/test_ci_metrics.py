from __future__ import annotations

from pathlib import Path

import pytest

from scripts import ci_metrics


def test_resolve_junit_path_prefers_explicit_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    junit_path = tmp_path / "pytest-junit.xml"
    junit_path.write_text("<testsuite></testsuite>", encoding="utf-8")

    resolved = ci_metrics.resolve_junit_path(junit_path)

    assert resolved == junit_path


def test_resolve_junit_path_falls_back_to_workspace_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "nested" / "pytest-junit.xml"
    nested.parent.mkdir(parents=True)
    nested.write_text("<testsuite></testsuite>", encoding="utf-8")

    resolved = ci_metrics.resolve_junit_path(Path("pytest-junit.xml"))

    assert resolved.resolve() == nested


def test_resolve_junit_path_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        ci_metrics.resolve_junit_path(Path("pytest-junit.xml"))
