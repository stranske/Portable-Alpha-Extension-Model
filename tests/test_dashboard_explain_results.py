from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import dashboard.components.explain_results as explain_module


class _FakeContext:
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class _DownloadCall:
    label: str
    data: str
    file_name: str
    mime: str


class FakeStreamlit:
    def __init__(self, *, button_value: bool) -> None:
        self.session_state: dict[str, Any] = {}
        self.button_value = button_value
        self.downloads: list[_DownloadCall] = []
        self.messages: list[tuple[str, str]] = []

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def text_area(self, label: str, value: str, key: str, help: str) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        self.messages.append(("expander", label))
        return _FakeContext()

    def selectbox(self, label: str, options: list[str], index: int, key: str) -> str:
        self.session_state.setdefault(key, options[index])
        return str(self.session_state[key])

    def text_input(self, label: str, value: str, key: str, type: str | None = None, help: str | None = None) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def button(self, label: str, key: str | None = None) -> bool:
        return self.button_value

    def spinner(self, text: str) -> _FakeContext:
        self.messages.append(("spinner", text))
        return _FakeContext()

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def markdown(self, text: str) -> None:
        self.messages.append(("markdown", text))

    def caption(self, text: str) -> None:
        self.messages.append(("caption", text))

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def download_button(self, label: str, data: str, file_name: str, mime: str) -> None:
        self.downloads.append(
            _DownloadCall(label=label, data=data, file_name=file_name, mime=mime)
        )


def test_render_explain_results_caches_and_download_json_fields(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(
            provider_name=kwargs["provider"],
            model_name=kwargs.get("model") or "gpt-4o-mini",
        ),
    )

    call_count = {"count": 0}

    def _fake_explain(summary_df: pd.DataFrame, manifest: dict[str, Any] | None):
        call_count["count"] += 1
        return "Explanation text", "https://smith.langchain.com/r/trace123", {"rows": len(summary_df)}

    monkeypatch.setattr(explain_module, "explain_results_details", _fake_explain)

    summary_df = pd.DataFrame({"monthly_TE": [0.01, 0.02], "monthly_CVaR": [-0.03, -0.02]})
    manifest = {"seed": 123}

    explain_module.render_explain_results_panel(
        summary_df=summary_df,
        manifest=manifest,
        xlsx_path="/tmp/Outputs.xlsx",
    )
    explain_module.render_explain_results_panel(
        summary_df=summary_df,
        manifest=manifest,
        xlsx_path="/tmp/Outputs.xlsx",
    )

    assert call_count["count"] == 1

    json_downloads = [item for item in fake_st.downloads if item.file_name.endswith(".json")]
    assert json_downloads
    payload = json.loads(json_downloads[0].data)
    assert "trace_url" in payload
    assert "created_at" in payload
    assert "inputs_summary" in payload
    assert payload["inputs_summary"]["manifest_seed"] == 123
    assert "api_key" not in payload["inputs_summary"]


def test_results_page_llm_fallback_message(monkeypatch) -> None:
    module = runpy.run_path(str(Path("dashboard/pages/4_Results.py")))

    class _PageFakeSt:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str) -> None:
            self.messages.append(message)

    fake_st = _PageFakeSt()

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError("langchain_openai")

    module["_render_explain_results"].__globals__["render_explain_results_panel"] = _raise
    module["_render_explain_results"].__globals__["st"] = fake_st

    module["_render_explain_results"](
        summary=pd.DataFrame({"x": [1.0]}),
        manifest_data={"seed": 1},
        xlsx="Outputs.xlsx",
    )

    assert any("LLM features unavailable" in message for message in fake_st.messages)
    assert any("install .[llm]" in message.lower() for message in fake_st.messages)
