from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _StopCalled(Exception):
    pass


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlit(ModuleType):
    def __init__(self, name: str, select_value: str = "run-1"):
        super().__init__(name)
        self.calls: list[tuple[str, object]] = []
        self._select_value = select_value

    def set_page_config(self, **kwargs) -> None:
        self.calls.append(("set_page_config", kwargs))

    def title(self, message: str) -> None:
        self.calls.append(("title", message))

    def info(self, message: str) -> None:
        self.calls.append(("info", message))

    def warning(self, message: str) -> None:
        self.calls.append(("warning", message))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))

    def subheader(self, message: str) -> None:
        self.calls.append(("subheader", message))

    def code(self, payload: str, language: str | None = None) -> None:
        self.calls.append(("code", payload))

    def text(self, payload: str) -> None:
        self.calls.append(("text", payload))

    def write(self, *args, **kwargs) -> None:
        self.calls.append(("write", args))

    def stop(self) -> None:
        raise _StopCalled("stop called")

    def selectbox(self, label: str, options: list[str]):
        self.calls.append(("selectbox", options))
        return self._select_value

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return [_FakeColumn() for _ in range(count)]


def _page_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dashboard/pages/7_Run_Logs.py"


def test_run_logs_no_runs_directory(monkeypatch, tmp_path: Path) -> None:
    fake_st = FakeStreamlit("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(_StopCalled):
        runpy.run_path(str(_page_path()))

    assert any(
        call == ("info", "No runs directory found yet. Launch a run with --log-json to create logs.")
        for call in fake_st.calls
    )


def test_run_logs_no_run_ids(monkeypatch, tmp_path: Path) -> None:
    fake_st = FakeStreamlit("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(_StopCalled):
        runpy.run_path(str(_page_path()))

    assert any(call == ("info", "No run directories available.") for call in fake_st.calls)


def test_run_logs_with_log_and_manifest(monkeypatch, tmp_path: Path) -> None:
    fake_st = FakeStreamlit("streamlit", select_value="run-1")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text('{"event":"ok"}\nplain line\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"version":"1"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(_page_path()))

    call_types = {call[0] for call in fake_st.calls}
    assert "code" in call_types
    assert "text" in call_types
    assert any(call[0] == "write" for call in fake_st.calls)


def test_run_logs_missing_log_and_manifest(monkeypatch, tmp_path: Path) -> None:
    fake_st = FakeStreamlit("streamlit", select_value="run-1")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(_page_path()))

    assert any(call[0] == "warning" for call in fake_st.calls)
    assert any(call[0] == "info" for call in fake_st.calls)
