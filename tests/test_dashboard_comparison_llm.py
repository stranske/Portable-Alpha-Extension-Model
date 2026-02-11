from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import dashboard.components.comparison_llm as comparison_module


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

    def caption(self, message: str) -> None:
        self.messages.append(("caption", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def markdown(self, message: str) -> None:
        self.messages.append(("markdown", message))

    def text_area(self, label: str, value: str, key: str, help: str) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        self.messages.append(("expander", label))
        return _FakeContext()

    def selectbox(self, label: str, options: list[str], index: int, key: str) -> str:
        self.session_state.setdefault(key, options[index])
        return str(self.session_state[key])

    def text_input(
        self, label: str, value: str, key: str, type: str | None = None, help: str | None = None
    ) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def button(self, label: str, key: str | None = None) -> bool:
        return self.button_value

    def spinner(self, text: str) -> _FakeContext:
        self.messages.append(("spinner", text))
        return _FakeContext()

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def download_button(self, label: str, data: str, file_name: str, mime: str) -> None:
        self.downloads.append(_DownloadCall(label=label, data=data, file_name=file_name, mime=mime))


def test_render_comparison_llm_panel_shows_default_message(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=False)
    monkeypatch.setattr(comparison_module, "st", fake_st)

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key="run-a-vs-run-b",
    )

    assert "comparison_llm_cache" in fake_st.session_state
    assert any(kind == "subheader" and "LLM Comparison" in text for kind, text in fake_st.messages)
    assert any(kind == "info" and "Click Compare Runs" in text for kind, text in fake_st.messages)


def test_render_comparison_llm_panel_generates_output_and_downloads(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(comparison_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(comparison_module, "resolve_llm_provider_config", lambda **kwargs: object())
    monkeypatch.setattr(
        comparison_module,
        "compare_runs",
        lambda **kwargs: (
            "Generated comparison",
            "https://smith.langchain.com/r/trace-id",
            comparison_module.CompareRunsPayload(
                config_diff="- seed: 1 -> 2",
                metric_catalog_a={"monthly_TE": 0.02},
                metric_catalog_b={"monthly_TE": 0.01},
                prompt="prompt",
                questions="questions",
                prior_manifest_path="prior_manifest.json",
                prior_summary_path="prior.xlsx",
            ),
        ),
    )
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda manifest_data: ({}, None))

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key="run-a-vs-run-b",
    )

    assert any(
        kind == "markdown" and "Generated comparison" in text for kind, text in fake_st.messages
    )
    assert any(kind == "caption" and "Trace URL" in text for kind, text in fake_st.messages)
    assert any(item.label == "Download TXT" for item in fake_st.downloads)
    assert any(item.label == "Download JSON" for item in fake_st.downloads)

    txt_export = next(item for item in fake_st.downloads if item.label == "Download TXT")
    assert "Config Diff" in txt_export.data
    assert "- seed: 1 -> 2" in txt_export.data
    assert "Trace URL: https://smith.langchain.com/r/trace-id" in txt_export.data

    json_export = next(item for item in fake_st.downloads if item.label == "Download JSON")
    assert '"config_diff": "- seed: 1 -> 2"' in json_export.data
    assert '"trace_url": "https://smith.langchain.com/r/trace-id"' in json_export.data
