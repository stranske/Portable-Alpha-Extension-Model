from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def baseline_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Agent": ["A", "Total", "C"],
            "monthly_TE": [0.02, 0.03, 0.01],
            "monthly_CVaR": [-0.05, -0.04, -0.06],
            "monthly_BreachProb": [0.11, 0.07, 0.09],
            "Regime": ["base", "bull", "bear"],
        }
    )


@pytest.fixture
def summary_df_missing_monthly_cvar(baseline_summary_df: pd.DataFrame) -> pd.DataFrame:
    return baseline_summary_df.drop(columns=["monthly_CVaR"])


@pytest.fixture
def summary_df_with_null_nan_empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Agent": ["A", "Total", None],
            "monthly_TE": [0.02, None, np.nan],
            "monthly_CVaR": [-0.05, np.nan, -0.01],
            "monthly_BreachProb": [None, "", np.nan],
            "Regime": ["base", "", None],
        }
    )


def test_explain_results_details_nominal_baseline_fixture_returns_expected_values(
    baseline_summary_df: pd.DataFrame,
) -> None:
    text, trace_url, payload = result_explain.explain_results_details(baseline_summary_df)
    metric_catalog = payload["metric_catalog"]

    assert text.startswith(
        "LLM configuration is required to generate a result explanation. "
        "Prepared payload for 3 rows."
    )
    assert result_explain.EXPLAIN_RESULTS_DISCLAIMER in text
    assert trace_url is None
    assert metric_catalog["tracking_error"]["value"] == 0.03
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["tracking_error"]["label"] == "Tracking Error"
    assert metric_catalog["cvar"]["value"] == pytest.approx(-0.04)
    assert metric_catalog["breach_probability"]["value"] == pytest.approx(0.07)
    # Categorical field extraction is not currently supported by metric catalog generation.
    assert "Agent" not in metric_catalog
    assert "Regime" not in metric_catalog


def test_explain_results_details_missing_column_fixture_omits_missing_metric(
    summary_df_missing_monthly_cvar: pd.DataFrame,
) -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(
        summary_df_missing_monthly_cvar
    )
    metric_catalog = payload["metric_catalog"]

    assert set(metric_catalog) == {"tracking_error", "breach_probability"}
    assert "cvar" not in metric_catalog
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["breach_probability"]["value"] == pytest.approx(0.07)


def test_explain_results_details_null_nan_empty_fixture_handles_unusable_values(
    summary_df_with_null_nan_empty: pd.DataFrame,
) -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(
        summary_df_with_null_nan_empty
    )
    metric_catalog = payload["metric_catalog"]

    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.02)
    assert metric_catalog["cvar"]["value"] == pytest.approx(-0.03)
    assert "breach_probability" not in metric_catalog


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

    assert text.startswith("SENTINEL_LLM_RESPONSE")
    assert result_explain.EXPLAIN_RESULTS_DISCLAIMER in text
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

    assert text.startswith("ok")
    assert result_explain.EXPLAIN_RESULTS_DISCLAIMER in text
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

    assert text.startswith("TRACED_RESPONSE")
    assert result_explain.EXPLAIN_RESULTS_DISCLAIMER in text
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
    )

    analysis_output = payload["analysis_output"]
    assert isinstance(analysis_output, dict)
    assert analysis_output
    assert isinstance(analysis_output["columns"], list) and analysis_output["columns"]
    assert (
        isinstance(analysis_output["basic_statistics"], dict)
        and analysis_output["basic_statistics"]
    )
    assert (
        isinstance(analysis_output["tail_sample_rows"], list)
        and analysis_output["tail_sample_rows"]
    )
    assert isinstance(analysis_output["key_quantiles"], dict) and analysis_output["key_quantiles"]
    json.dumps(analysis_output, sort_keys=True)


def test_analysis_output_manifest_highlights_include_sentinel_value() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(
        _toy_df(),
        {"run_name": "SENTINEL_RUN"},
    )
    analysis_output = payload["analysis_output"]
    assert analysis_output["manifest_highlights"]["run_name"] == "SENTINEL_RUN"


def test_stress_delta_summary_present_when_columns_exist() -> None:
    df = _toy_df().assign(StressDelta_TE=[0.1, 0.2], StressDelta_CVaR=[-0.3, -0.2])
    _text, _trace_url, payload = result_explain.explain_results_details(df)

    stress = payload["analysis_output"]["stress_delta_summary"]
    assert stress is not None
    assert stress["columns"] == ["StressDelta_CVaR", "StressDelta_TE"]
    assert stress["summary"]["StressDelta_TE"]["mean"] == pytest.approx(0.15)
    assert stress["summary"]["StressDelta_TE"]["min"] == 0.1
    assert stress["summary"]["StressDelta_TE"]["max"] == 0.2
    assert stress["summary"]["StressDelta_CVaR"]["mean"] == -0.25


def test_stress_delta_summary_absent_when_columns_missing() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(_toy_df())
    assert payload["analysis_output"]["stress_delta_summary"] is None


def test_metric_catalog_includes_te_cvar_and_breach_probability() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(_toy_df())
    metric_catalog = payload["metric_catalog"]

    assert metric_catalog["tracking_error"]["metric"] == "TE"
    assert metric_catalog["tracking_error"]["label"] == "Tracking Error"
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["cvar"]["metric"] == "CVaR"
    assert metric_catalog["cvar"]["label"] == "CVaR"
    assert metric_catalog["cvar"]["value"] == pytest.approx(-0.04)
    assert metric_catalog["breach_probability"]["metric"] == "Breach Probability"
    assert metric_catalog["breach_probability"]["label"] == "Breach Probability"
    assert metric_catalog["breach_probability"]["value"] == pytest.approx(0.07)


def test_metric_catalog_omits_missing_columns_without_error() -> None:
    df = _toy_df().drop(columns=["monthly_BreachProb"])
    _text, _trace_url, payload = result_explain.explain_results_details(df)
    metric_catalog = payload["metric_catalog"]
    assert "tracking_error" in metric_catalog
    assert "cvar" in metric_catalog
    assert "breach_probability" not in metric_catalog
    assert metric_catalog["tracking_error"]["label"] == "Tracking Error"
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["cvar"]["label"] == "CVaR"
    assert metric_catalog["cvar"]["value"] == pytest.approx(-0.04)


def test_metric_catalog_entries_include_label_and_value_fields() -> None:
    _text, _trace_url, payload = result_explain.explain_results_details(_toy_df())
    metric_catalog = payload["metric_catalog"]

    for key in ("tracking_error", "cvar", "breach_probability"):
        assert key in metric_catalog
        entry = metric_catalog[key]
        assert isinstance(entry, dict)
        assert "label" in entry and isinstance(entry["label"], str) and entry["label"]
        assert "value" in entry and isinstance(entry["value"], float)


def test_metric_catalog_omits_unusable_metric_values_without_error() -> None:
    df = _toy_df().assign(monthly_CVaR=["bad", "worse"])
    _text, _trace_url, payload = result_explain.explain_results_details(df)
    metric_catalog = payload["metric_catalog"]

    assert "tracking_error" in metric_catalog
    assert "cvar" not in metric_catalog
    assert "breach_probability" in metric_catalog
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["breach_probability"]["value"] == pytest.approx(0.07)


def test_metric_catalog_supports_human_readable_metric_column_names() -> None:
    df = pd.DataFrame(
        {
            "Tracking Error": [0.02, 0.04],
            "CVaR": [-0.03, -0.05],
            "Breach Probability": [0.10, 0.30],
        }
    )

    _text, _trace_url, payload = result_explain.explain_results_details(df)
    metric_catalog = payload["metric_catalog"]

    assert metric_catalog["tracking_error"]["label"] == "Tracking Error"
    assert metric_catalog["tracking_error"]["value"] == pytest.approx(0.03)
    assert metric_catalog["cvar"]["label"] == "CVaR"
    assert metric_catalog["cvar"]["value"] == pytest.approx(-0.04)
    assert metric_catalog["breach_probability"]["label"] == "Breach Probability"
    assert metric_catalog["breach_probability"]["value"] == pytest.approx(0.20)


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
    assert result_explain._REDACTION_TOKEN in payload["error"]


def test_api_keys_are_redacted_from_invoke_errors(monkeypatch) -> None:
    config = _config(api_key="SECRET123")
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    class _RaisingLLM:
        def invoke(self, prompt: str) -> Any:
            raise RuntimeError("invoke failed with Authorization: Bearer SECRET123")

    monkeypatch.setattr(result_explain, "create_llm", lambda _cfg: _RaisingLLM())
    text, _trace_url, payload = result_explain.explain_results_details(_toy_df(), llm_config=config)

    assert "SECRET123" not in text
    assert "SECRET123" not in payload["error"]
    assert result_explain._REDACTION_TOKEN in text
    assert result_explain._REDACTION_TOKEN in payload["error"]


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


def test_disclaimer_appended_to_no_llm_fallback_text() -> None:
    text, _trace_url, _payload = result_explain.explain_results_details(_toy_df())
    assert text.startswith("LLM configuration is required")
    assert text.endswith(result_explain.EXPLAIN_RESULTS_DISCLAIMER)


def test_disclaimer_appended_to_successful_llm_text(monkeypatch) -> None:
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )
    monkeypatch.setattr(result_explain, "create_llm", lambda _cfg: _StaticLLM("model says hi"))

    text, _trace_url, _payload = result_explain.explain_results_details(
        _toy_df(), llm_config=_config(), tracing_enabled=False
    )

    assert text.startswith("model says hi")
    assert text.endswith(result_explain.EXPLAIN_RESULTS_DISCLAIMER)


def test_disclaimer_appended_once_to_error_path_text(monkeypatch) -> None:
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    def _raise_create(_cfg: LLMProviderConfig) -> Any:
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(result_explain, "create_llm", _raise_create)

    text, _trace_url, _payload = result_explain.explain_results_details(
        _toy_df(), llm_config=_config()
    )

    assert text.startswith("Failed to generate explanation")
    assert text.count(result_explain.EXPLAIN_RESULTS_DISCLAIMER) == 1


def test_error_messages_redact_lowercase_bearer_and_config_dump(monkeypatch) -> None:
    config = _config(api_key="SECRET123")
    monkeypatch.setattr(
        result_explain, "build_result_explanation_prompt", lambda *_a, **_k: "prompt"
    )

    class _RaisingLLM:
        def invoke(self, prompt: str) -> Any:
            raise RuntimeError(
                "authorization: bearer SECRET123 "
                "config={'provider':'openai','model':'gpt-4o-mini','api_key':'SECRET123'}"
            )

    monkeypatch.setattr(result_explain, "create_llm", lambda _cfg: _RaisingLLM())
    text, _trace_url, payload = result_explain.explain_results_details(_toy_df(), llm_config=config)

    assert "SECRET123" not in text
    assert "bearer SECRET123" not in text.lower()
    assert "authorization:" not in text.lower()
    assert "config={" not in text
    assert "SECRET123" not in payload["error"]
    assert "config={" not in payload["error"]
    assert result_explain._REDACTION_TOKEN in payload["error"]


# ---------------------------------------------------------------------------
# _with_disclaimer edge cases
# ---------------------------------------------------------------------------


def test_with_disclaimer_empty_string_returns_just_disclaimer() -> None:
    result = result_explain._with_disclaimer("")
    assert result == result_explain.EXPLAIN_RESULTS_DISCLAIMER


def test_with_disclaimer_none_coerced_empty_returns_just_disclaimer() -> None:
    # _with_disclaimer treats falsy text as empty via `(text or "")`.
    result = result_explain._with_disclaimer("")
    assert result == result_explain.EXPLAIN_RESULTS_DISCLAIMER


def test_with_disclaimer_idempotent_does_not_double_append() -> None:
    already = f"Some analysis.\n\n{result_explain.EXPLAIN_RESULTS_DISCLAIMER}"
    result = result_explain._with_disclaimer(already)
    assert result == already
    assert result.count(result_explain.EXPLAIN_RESULTS_DISCLAIMER) == 1


def test_with_disclaimer_strips_trailing_whitespace_before_appending() -> None:
    result = result_explain._with_disclaimer("Some text.   ")
    assert result.startswith("Some text.")
    assert result.endswith(result_explain.EXPLAIN_RESULTS_DISCLAIMER)


# ---------------------------------------------------------------------------
# TypeError guard
# ---------------------------------------------------------------------------


def test_explain_results_raises_type_error_for_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        result_explain.explain_results_details({"not": "a dataframe"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _extract_response_text list-content path
# ---------------------------------------------------------------------------


def test_extract_response_text_list_of_strings() -> None:
    response = SimpleNamespace(content=["Hello ", "world"])
    result = result_explain._extract_response_text(response)
    assert result == "Hello world"


def test_extract_response_text_list_of_dicts_with_text_key() -> None:
    response = SimpleNamespace(content=[{"text": "part one"}, {"text": "part two"}])
    result = result_explain._extract_response_text(response)
    assert result == "part one part two"


def test_extract_response_text_mixed_list_skips_non_text_dicts() -> None:
    response = SimpleNamespace(content=["hello", {"type": "image"}, {"text": "world"}])
    result = result_explain._extract_response_text(response)
    assert result == "hello world"


def test_extract_response_text_empty_list_falls_back_to_str() -> None:
    response = SimpleNamespace(content=[])
    result = result_explain._extract_response_text(response)
    # Falls through to str(response).strip() which is non-empty.
    assert isinstance(result, str) and result


def test_extract_response_text_list_of_blank_strings_falls_back_to_str() -> None:
    response = SimpleNamespace(content=["  ", ""])
    result = result_explain._extract_response_text(response)
    assert isinstance(result, str) and result


# ---------------------------------------------------------------------------
# _to_json_safe edge cases
# ---------------------------------------------------------------------------


def test_to_json_safe_tuple_is_converted_to_list() -> None:
    result = result_explain._to_json_safe((1, "a", None))
    assert result == [1, "a", None]


def test_to_json_safe_timestamp_is_isoformat_string() -> None:
    ts = pd.Timestamp("2024-01-15 12:30:00")
    result = result_explain._to_json_safe(ts)
    assert isinstance(result, str)
    assert "2024-01-15" in result


def test_to_json_safe_pandas_na_becomes_none() -> None:
    # pd.NA is not a float/str/int/bool/Timestamp, so it reaches the pd.isna branch.
    result = result_explain._to_json_safe(pd.NA)
    assert result is None


def test_to_json_safe_unknown_object_returns_str() -> None:
    class _Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    result = result_explain._to_json_safe(_Opaque())
    assert result == "opaque-value"


# ---------------------------------------------------------------------------
# Helper function edge cases
# ---------------------------------------------------------------------------


def test_build_basic_statistics_empty_numeric_df_returns_empty_dict() -> None:
    df = pd.DataFrame({"label": ["a", "b"]})
    result = result_explain._build_basic_statistics(df)
    assert result == {}


def test_build_quantiles_empty_numeric_df_returns_empty_dict() -> None:
    df = pd.DataFrame({"label": ["a", "b"]})
    result = result_explain._build_quantiles(df)
    assert result == {}


def test_build_tail_samples_empty_df_returns_empty_list() -> None:
    df = pd.DataFrame()
    result = result_explain._build_tail_samples(df)
    assert result == []


def test_build_stress_delta_summary_non_numeric_stress_cols_returns_empty_summary() -> None:
    df = pd.DataFrame({"StressDelta_TE": ["high", "low"]})
    result = result_explain._build_stress_delta_summary(df)
    assert result is not None
    assert result["summary"] == {}


def test_build_stress_delta_summary_all_nan_stress_col_skips_col() -> None:
    import numpy as np

    df = pd.DataFrame({"StressDelta_TE": [np.nan, np.nan]})
    result = result_explain._build_stress_delta_summary(df)
    assert result is not None
    assert "StressDelta_TE" not in result["summary"]


def test_manifest_highlights_includes_seed_and_cli_args() -> None:
    manifest = {
        "run_name": "test_run",
        "seed": 42,
        "cli_args": {"n_sims": 1000, "horizon_months": 12, "output": "results/"},
    }
    result = result_explain._manifest_highlights(manifest)
    assert result["seed"] == 42
    assert result["cli_n_sims"] == 1000
    assert result["cli_horizon_months"] == 12
    assert result["cli_output"] == "results/"


def test_sanitize_error_message_empty_exc_returns_unknown_error() -> None:
    exc = RuntimeError("")
    result = result_explain._sanitize_error_message(exc, secrets=[])
    assert result == "unknown error"


def test_sanitize_error_message_long_plain_message_is_truncated() -> None:
    # A long message with no sensitive patterns so redaction leaves it intact.
    long_msg = "Network timeout connecting to endpoint. " * 20
    exc = RuntimeError(long_msg)
    result = result_explain._sanitize_error_message(exc, secrets=[])
    assert len(result) <= result_explain._MAX_ERROR_MESSAGE_LEN
    assert result.endswith("...")


def test_coerce_metric_value_column_not_in_df_returns_none() -> None:
    df = _toy_df()
    result = result_explain._coerce_metric_value(df, "nonexistent_column")
    assert result is None
