"""Result explanation entrypoint for dashboard Results details."""

from __future__ import annotations

import os
import re
from contextlib import nullcontext
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

import pandas as pd

from pa_core.llm.prompts import build_result_explanation_prompt
from pa_core.llm.provider import LLMProviderConfig, create_llm
from pa_core.llm.tracing import langsmith_tracing_context, resolve_trace_url

_REDACTION_TOKEN = "[REDACTED]"
_MAX_ERROR_MESSAGE_LEN = 500
_TAIL_ROW_LIMIT = 3
_QUANTILE_LEVELS = [0.05, 0.5, 0.95]
_METRIC_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "tracking_error": (
        "monthly_te",
        "te",
        "tracking error",
        "trackingerror",
        "trackingerr",
        "tracking_error",
        "monthly_tracking_error",
    ),
    "cvar": ("monthly_cvar", "cvar", "terminal_cvar", "cvar95"),
    "breach_probability": (
        "monthly_breachprob",
        "breachprob",
        "breach_probability",
        "breachprobability",
        "breach_prob",
    ),
}
_STRESS_DELTA_MARKERS = (
    "stressdelta",
    "stress_delta",
    "te_delta",
    "cvar_delta",
    "breach_delta",
    "delta_te",
    "delta_cvar",
    "delta_breach",
)


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_metric_column(details_df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    by_normalized: dict[str, str] = {
        _normalize_column_name(str(col)): str(col) for col in details_df.columns
    }
    for alias in aliases:
        resolved = by_normalized.get(_normalize_column_name(alias))
        if resolved:
            return resolved
    return None


def _coerce_metric_value(details_df: pd.DataFrame, column: str) -> float | None:
    if column not in details_df.columns:
        return None
    numeric = pd.to_numeric(details_df[column], errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return None

    if "Agent" in details_df.columns:
        agent = details_df["Agent"].astype(str).str.strip().str.lower()
        total_match = valid[agent == "total"]
        if not total_match.empty:
            return float(total_match.iloc[0])
    return float(valid.mean())


def _build_metric_catalog(details_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    labels = {
        "tracking_error": "Tracking Error",
        "cvar": "CVaR",
        "breach_probability": "Breach Probability",
    }
    metric_codes = {
        "tracking_error": "TE",
        "cvar": "CVaR",
        "breach_probability": "Breach Probability",
    }
    for metric_key, aliases in _METRIC_ALIAS_GROUPS.items():
        col_name = _find_metric_column(details_df, aliases)
        if not col_name:
            continue
        metric_value = _coerce_metric_value(details_df, col_name)
        if metric_value is None:
            continue
        catalog[metric_key] = {
            "metric": metric_codes[metric_key],
            "label": labels[metric_key],
            "value": metric_value,
            "column": col_name,
        }
    return catalog


def _build_basic_statistics(details_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric_df = details_df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    stat_rows = numeric_df.agg(["count", "mean", "std", "min", "max"]).to_dict()
    return cast(dict[str, dict[str, float]], _to_json_safe(stat_rows))


def _build_quantiles(details_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric_df = details_df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    quantiles = numeric_df.quantile(_QUANTILE_LEVELS).to_dict()
    return cast(dict[str, dict[str, float]], _to_json_safe(quantiles))


def _build_tail_samples(details_df: pd.DataFrame) -> list[dict[str, Any]]:
    if details_df.empty:
        return []
    tail_df = details_df.tail(_TAIL_ROW_LIMIT).copy()
    return cast(list[dict[str, Any]], _to_json_safe(tail_df.to_dict(orient="records")))


def _build_stress_delta_summary(details_df: pd.DataFrame) -> dict[str, Any] | None:
    relevant_cols = [
        str(col)
        for col in details_df.columns
        if any(marker in _normalize_column_name(str(col)) for marker in _STRESS_DELTA_MARKERS)
    ]
    if not relevant_cols:
        return None
    # Keep output stable even if upstream dataframe column order changes.
    relevant_cols = sorted(relevant_cols, key=_normalize_column_name)
    numeric_subset = details_df[relevant_cols].select_dtypes(include="number")
    if numeric_subset.empty:
        return {"columns": relevant_cols, "summary": {}}
    summary: dict[str, dict[str, float]] = {}
    for col in numeric_subset.columns:
        series = pd.to_numeric(numeric_subset[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[str(col)] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return {"columns": relevant_cols, "summary": summary}


def _manifest_highlights(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    highlights: dict[str, Any] = {}
    run_name = manifest.get("run_name")
    if run_name is not None:
        highlights["run_name"] = run_name
    seed = manifest.get("seed")
    if seed is not None:
        highlights["seed"] = seed
    cli_args = manifest.get("cli_args")
    if isinstance(cli_args, Mapping):
        # Keep this compact and avoid leaking any key-like values.
        for key in ("output", "benchmark", "config", "n_sims", "horizon_months"):
            value = cli_args.get(key)
            if value is not None:
                highlights[f"cli_{key}"] = value
    return highlights


def _redact_sensitive(text: str, *, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if isinstance(secret, str) and secret:
            redacted = redacted.replace(secret, _REDACTION_TOKEN)
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", _REDACTION_TOKEN, redacted)
    redacted = re.sub(
        r"Authorization\s*:\s*[^\s,;]+",
        _REDACTION_TOKEN,
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(api_?key|token|credential)s?\s*=\s*[^,\s]+",
        _REDACTION_TOKEN,
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"credentials\s*=\s*\{[^}]*\}",
        _REDACTION_TOKEN,
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"['\"]?(api_?key|authorization|bearer|token)['\"]?\s*:\s*['\"][^'\"]*['\"]",
        _REDACTION_TOKEN,
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def _sanitize_error_message(exc: Exception, *, secrets: Sequence[str]) -> str:
    cleaned = _redact_sensitive(str(exc), secrets=secrets)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > _MAX_ERROR_MESSAGE_LEN:
        cleaned = cleaned[: _MAX_ERROR_MESSAGE_LEN - 3] + "..."
    return cleaned or "unknown error"


def _is_tracing_enabled(tracing_enabled: bool | None) -> bool:
    if tracing_enabled is not None:
        return tracing_enabled
    return bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def _extract_response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        joined = " ".join(part.strip() for part in chunks if part.strip()).strip()
        if joined:
            return joined
    return str(response).strip()


def explain_results_details(
    details_df: pd.DataFrame,
    manifest: Mapping[str, Any] | None = None,
    *,
    questions: Sequence[str] | str | None = None,
    llm_config: LLMProviderConfig | None = None,
    tracing_enabled: bool | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Return an explain-results payload for a summary/details dataframe.

    Parameters
    ----------
    details_df
        Results details table (typically the Summary sheet).
    manifest
        Optional run manifest payload.

    Returns
    -------
    tuple[str, str | None, dict[str, Any]]
        ``(text, trace_url, payload_dict)``.
    """

    if not isinstance(details_df, pd.DataFrame):
        raise TypeError("details_df must be a pandas DataFrame")

    analysis_output: dict[str, Any] = {
        "rows": int(details_df.shape[0]),
        "columns": [str(col) for col in details_df.columns.tolist()],
        "basic_statistics": _build_basic_statistics(details_df),
        "tail_sample_rows": _build_tail_samples(details_df),
        "key_quantiles": _build_quantiles(details_df),
        "manifest_highlights": _manifest_highlights(manifest),
        "stress_delta_summary": _build_stress_delta_summary(details_df),
    }
    metric_catalog = _build_metric_catalog(details_df)
    payload: dict[str, Any] = {
        "analysis_output": analysis_output,
        "metric_catalog": metric_catalog,
        "trace_url": None,
    }

    prompt_input = {
        "analysis_output": analysis_output,
        "metric_catalog": metric_catalog,
    }

    config = llm_config
    if config is None:
        # Keep this deterministic for direct calls without dashboard wiring.
        text = (
            "LLM configuration is required to generate a result explanation. "
            f"Prepared payload for {analysis_output['rows']} rows."
        )
        return text, None, payload

    api_key = config.credentials.get("api_key", "")
    request_id = uuid4().hex
    trace_url: str | None = None

    try:
        prompt = build_result_explanation_prompt(prompt_input, questions=questions)
        llm = create_llm(config)
        use_tracing = _is_tracing_enabled(tracing_enabled)
        tracing_cm = (
            langsmith_tracing_context(
                project_name="portable-alpha-explain-results",
                tags=["explain_results"],
                metadata={
                    "request_id": request_id,
                    "provider": config.provider_name,
                    "model": config.model_name,
                },
            )
            if use_tracing
            else nullcontext()
        )
        with tracing_cm:
            response = llm.invoke(prompt)
        text = _extract_response_text(response)
        if use_tracing:
            trace_url = resolve_trace_url(request_id)
            payload["trace_url"] = trace_url
    except Exception as exc:  # pragma: no cover - behavior validated by tests
        safe_error = _sanitize_error_message(exc, secrets=[api_key])
        text = f"Failed to generate explanation: {safe_error}"
        payload["error"] = safe_error
        trace_url = None
    return text, trace_url, payload
