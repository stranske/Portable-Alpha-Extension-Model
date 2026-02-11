from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd

import pa_core.llm.result_explain as result_explain
from pa_core.llm.provider import LLMProviderConfig


class _FakeContext:
    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def __enter__(self) -> "_FakeContext":
        self._calls["entered"] = self._calls.get("entered", 0) + 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._calls["exited"] = self._calls.get("exited", 0) + 1
        return False


class _StaticLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


class _ContextAwareLLM:
    def __init__(self, state: dict[str, bool], text: str = "ok") -> None:
        self._state = state
        self._text = text

    def invoke(self, prompt: str) -> Any:
        assert self._state.get("inside") is True
        return SimpleNamespace(content=self._text)


def _config(api_key: str = "sk-test") -> LLMProviderConfig:
    return LLMProviderConfig(
        provider_name="openai",
        credentials={"api_key": api_key},
        model_name="gpt-4o-mini",
    )


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Agent": ["A", "Total"],
            "monthly_TE": [0.02, 0.03],
            "monthly_CVaR": [-0.05, -0.04],
            "monthly_BreachProb": [0.11, 0.07],
        }
    )


def test_explain_results_calls_prompt_builder_and_llm_once(monkeypatch) -> None:
    prompt_calls: list[dict[str, Any]] = []
    llm = _StaticLLM("SENTINEL_LLM_RESPONSE")
    create_calls: list[LLMProviderConfig] = []

    def _fake_prompt_builder(result_data: Any, *, questions: Any = None) -> str:
        prompt_calls.append({"result_data": result_data, "questions": questions})
        return "PROMPT-SENTINEL"

    def _fake_create_llm(config: LLMProviderConfig) -> _StaticLLM:
        create_calls.append(config)
        return llm

    monkeypatch.setattr(result_explain, "build_result_explanation_prompt", _fake_prompt_builder)
    monkeypatch.setattr(result_explain, "create_llm", _fake_create_llm)

    text, trace_url, _payload = result_explain.explain_results_details(
        _toy_df(),
        {"run_name": "SENTINEL_RUN"},
        questions=["Why?"],
        llm_config=_config(),
        tracing_enabled=False,
    )

    assert text == "SENTINEL_LLM_RESPONSE"
    assert trace_url is None
    assert len(prompt_calls) == 1
    assert len(create_calls) == 1
    assert llm.prompts == ["PROMPT-SENTINEL"]
    assert "Result explanation is ready" not in text


def test_questions_parameter_is_in_signature_and_forwarded(monkeypatch) -> None:
    sig = inspect.signature(result_explain.explain_results_details)
    assert "questions" in sig.parameters

    captured_questions: list[Any] = []

    def _fake_prompt_builder(result_data: Any, *, questions: Any = None) -> str:
        captured_questions.append(questions)
        return "prompt"

    monkeypatch.setattr(result_explain, "build_result_explanation_prompt", _fake_prompt_builder)
    monkeypatch.setattr(result_explain, "create_llm", lambda config: _StaticLLM("ok"))

    result_explain.explain_results_details(
        _toy_df(),
        questions=["Q1", "Q2"],
        llm_config=_config(),
        tracing_enabled=False,
    )

    assert captured_questions == [["Q1", "Q2"]]


def test_questions_forwarded_exactly_as_list(monkeypatch) -> None:
    captured_call: dict[str, Any] = {}

    def _fake_prompt_builder(result_data: Any, *, questions: Any = None) -> str:
        captured_call["result_data"] = result_data
        captured_call["questions"] = questions
        return "prompt"

    monkeypatch.setattr(result_explain, "build_result_explanation_prompt", _fake_prompt_builder)
    monkeypatch.setattr(result_explain, "create_llm", lambda config: _StaticLLM("ok"))

    result_explain.explain_results_details(
        _toy_df(),
        manifest={"run_name": "SENTINEL_RUN"},
        questions=["Q1", "Q2"],
        llm_config=_config(),
        tracing_enabled=False,
    )

    assert captured_call["questions"] == ["Q1", "Q2"]
    assert "analysis_output" in captured_call["result_data"]
    assert "metric_catalog" in captured_call["result_data"]


def test_create_llm_receives_dashboard_config_and_api_key_not_exposed(monkeypatch) -> None:
    config = _config(api_key="SECRET123")
    captured: list[LLMProviderConfig] = []
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    def _fake_create_llm(cfg: LLMProviderConfig) -> _StaticLLM:
        captured.append(cfg)
        return _StaticLLM("ok")

    monkeypatch.setattr(result_explain, "create_llm", _fake_create_llm)

    text, _trace_url, payload = result_explain.explain_results_details(
        _toy_df(),
        llm_config=config,
        tracing_enabled=False,
    )

    assert text == "ok"
    assert captured[0].provider_name == "openai"
    assert captured[0].model_name == "gpt-4o-mini"
    assert "api_key" in captured[0].credentials
    dumped = json.dumps(payload, sort_keys=True)
    assert "SECRET123" not in dumped


def test_langsmith_tracing_enabled_returns_trace_url(monkeypatch) -> None:
    trace_calls: dict[str, int] = {}
    resolve_calls: list[str | None] = []

    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )
    monkeypatch.setattr(result_explain, "create_llm", lambda config: _StaticLLM("ok"))
    monkeypatch.setattr(
        result_explain,
        "langsmith_tracing_context",
        lambda **_kwargs: _FakeContext(trace_calls),
    )

    def _fake_resolve(trace_id: str | None) -> str:
        resolve_calls.append(trace_id)
        return "https://smith.langchain.com/r/mock-trace"

    monkeypatch.setattr(result_explain, "resolve_trace_url", _fake_resolve)

    _text, trace_url, payload = result_explain.explain_results_details(
        _toy_df(),
        llm_config=_config(),
        tracing_enabled=True,
    )

    assert trace_calls["entered"] == 1
    assert len(resolve_calls) == 1
    assert isinstance(resolve_calls[0], str) and resolve_calls[0]
    assert trace_url == "https://smith.langchain.com/r/mock-trace"
    assert payload["trace_url"] == "https://smith.langchain.com/r/mock-trace"


def test_langsmith_tracing_wraps_llm_invoke(monkeypatch) -> None:
    trace_state = {"inside": False}

    class _StatefulContext:
        def __enter__(self):
            trace_state["inside"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            trace_state["inside"] = False
            return False

    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )
    monkeypatch.setattr(
        result_explain,
        "create_llm",
        lambda _config: _ContextAwareLLM(trace_state, text="TRACED_RESPONSE"),
    )
    monkeypatch.setattr(
        result_explain,
        "langsmith_tracing_context",
        lambda **_kwargs: _StatefulContext(),
    )
    monkeypatch.setattr(
        result_explain,
        "resolve_trace_url",
        lambda _trace_id: "https://smith.langchain.com/r/context-verified",
    )

    text, trace_url, _payload = result_explain.explain_results_details(
        _toy_df(), llm_config=_config(), tracing_enabled=True
    )

    assert text == "TRACED_RESPONSE"
    assert trace_url == "https://smith.langchain.com/r/context-verified"
    assert trace_state["inside"] is False


def test_langsmith_tracing_disabled_skips_trace_resolution(monkeypatch) -> None:
    trace_calls = {"context": 0, "resolve": 0}

    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )
    monkeypatch.setattr(result_explain, "create_llm", lambda config: _StaticLLM("ok"))

    def _fake_context(**_kwargs):
        trace_calls["context"] += 1
        return _FakeContext({})

    def _fake_resolve(trace_id: str | None) -> str:
        trace_calls["resolve"] += 1
        return "https://smith.langchain.com/r/should-not-be-called"

    monkeypatch.setattr(result_explain, "langsmith_tracing_context", _fake_context)
    monkeypatch.setattr(result_explain, "resolve_trace_url", _fake_resolve)

    _text, trace_url, payload = result_explain.explain_results_details(
        _toy_df(),
        llm_config=_config(),
        tracing_enabled=False,
    )

    assert trace_calls["context"] == 0
    assert trace_calls["resolve"] == 0
    assert trace_url is None
    assert payload["trace_url"] is None


def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(
        _toy_df(),
        {"run_name": "SENTINEL_RUN"},
    )

    analysis_output = payload["analysis_output"]
    assert analysis_output["columns"]
    assert analysis_output["basic_statistics"]
    assert analysis_output["tail_sample_rows"]
    assert analysis_output["key_quantiles"]
    assert analysis_output["manifest_highlights"]["run_name"] == "SENTINEL_RUN"
    json.dumps(analysis_output, sort_keys=True)


def test_stress_delta_summary_present_when_columns_exist() -> None:
    df = _toy_df().assign(StressDelta_TE=[0.1, 0.2], StressDelta_CVaR=[-0.3, -0.2])
    _text, _trace_url, payload = result_explain.explain_results_details(df)

    stress = payload["analysis_output"]["stress_delta_summary"]
    assert stress is not None
    assert "StressDelta_TE" in stress["columns"]


def test_stress_delta_summary_absent_when_columns_missing() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(_toy_df())
    assert payload["analysis_output"]["stress_delta_summary"] is None


def test_metric_catalog_includes_te_cvar_and_breach_probability() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(_toy_df())
    metric_catalog = payload["metric_catalog"]

    assert metric_catalog["tracking_error"]["label"] == "Tracking Error"
    assert isinstance(metric_catalog["tracking_error"]["value"], float)
    assert metric_catalog["cvar"]["label"] == "CVaR"
    assert isinstance(metric_catalog["cvar"]["value"], float)
    assert metric_catalog["breach_probability"]["label"] == "Breach Probability"
    assert isinstance(metric_catalog["breach_probability"]["value"], float)


def test_metric_catalog_omits_missing_columns_without_error() -> None:
    df = _toy_df().drop(columns=["monthly_BreachProb"])
    _text, _trace_url, payload = result_explain.explain_results_details(df)
    metric_catalog = payload["metric_catalog"]
    assert "tracking_error" in metric_catalog
    assert "cvar" in metric_catalog
    assert "breach_probability" not in metric_catalog


def test_api_keys_are_redacted_from_error_messages(monkeypatch) -> None:
    config = _config(api_key="SECRET123")
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    def _raise_create(_cfg: LLMProviderConfig) -> Any:
        raise RuntimeError("llm failure using key SECRET123")

    monkeypatch.setattr(result_explain, "create_llm", _raise_create)
    text, _trace_url, payload = result_explain.explain_results_details(_toy_df(), llm_config=config)

    assert "SECRET123" not in text
    assert result_explain._REDACTION_TOKEN in text
    assert "SECRET123" not in payload["error"]


def test_error_messages_are_sanitized_and_bounded(monkeypatch) -> None:
    config = _config(api_key="SECRET456")
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    class _RaisingLLM:
        def invoke(self, prompt: str) -> Any:
            long_tail = "X" * 2000
            raise RuntimeError(
                f"Authorization: abc123 Bearer token123 credentials={{'api_key':'SECRET456'}} {long_tail}"
            )

    monkeypatch.setattr(result_explain, "create_llm", lambda _cfg: _RaisingLLM())
    text, _trace_url, payload = result_explain.explain_results_details(_toy_df(), llm_config=config)

    assert "Authorization:" not in text
    assert "Bearer token123" not in text
    assert "SECRET456" not in text
    assert len(payload["error"]) <= result_explain._MAX_ERROR_MESSAGE_LEN
    assert "credentials" not in payload["error"].lower()
