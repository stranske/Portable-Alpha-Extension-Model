"""Run-to-run comparison helpers for Results page LLM panel."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from pa_core.contracts import (
    SUMMARY_BREACH_PROB_COLUMN,
    SUMMARY_CVAR_COLUMN,
    SUMMARY_SHEET_NAME,
    SUMMARY_TE_COLUMN,
    SUMMARY_TRACKING_ERROR_LEGACY_COLUMN,
)
from pa_core.llm.prompts import build_comparison_prompt
from pa_core.llm.tracing import langsmith_tracing_context, resolve_trace_url

_CLI_DIFF_KEYS: tuple[str, ...] = (
    "capital",
    "total_capital",
    "weights",
    "distribution",
    "return_distribution",
    "vol_distribution",
)

_WIZARD_KEYS: tuple[str, ...] = (
    "wizard",
    "wizard_config",
    "wizard_inputs",
)


@dataclass(frozen=True)
class CompareRunsPayload:
    """Serializable payload used by the comparison panel exports."""

    config_diff: str
    metric_catalog_a: dict[str, float]
    metric_catalog_b: dict[str, float]
    prompt: str
    questions: str
    prior_manifest_path: str | None
    prior_summary_path: str | None


def load_prior_manifest(
    manifest_data: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the prior-run manifest referenced by ``manifest_data['previous_run']``.

    Returns ``(prior_manifest, prior_manifest_path)``. If no usable previous-run
    reference is present, returns ``(None, None)``. If a path is present but
    missing or not a file, returns ``(None, path)``.

    Notes
    -----
    This function intentionally raises read/parse errors (for example
    ``PermissionError`` and ``json.JSONDecodeError``) so callers can decide how
    to surface unreadable artifact details.
    """

    if manifest_data is None:
        return None, None

    prev_ref = manifest_data.get("previous_run")
    if not isinstance(prev_ref, str) or not prev_ref.strip():
        return None, None

    prev_manifest_path = Path(prev_ref).expanduser()
    if not prev_manifest_path.exists() or not prev_manifest_path.is_file():
        return None, prev_manifest_path

    loaded = json.loads(prev_manifest_path.read_text())
    if not isinstance(loaded, dict):
        return None, prev_manifest_path
    return loaded, prev_manifest_path


def load_prior_summary(
    prior_manifest: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame | None, Path | None]:
    """Load prior Summary sheet referenced by prior manifest output path."""

    if not isinstance(prior_manifest, Mapping):
        return None, None
    cli_args = prior_manifest.get("cli_args")
    if not isinstance(cli_args, Mapping):
        return None, None
    output = cli_args.get("output")
    if not isinstance(output, str) or not output.strip():
        return None, None

    output_path = Path(output).expanduser()
    if not output_path.exists() or not output_path.is_file():
        return None, output_path

    frame = pd.read_excel(output_path, sheet_name=SUMMARY_SHEET_NAME)
    return frame, output_path


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _diff_line(label: str, old: Any, new: Any) -> str:
    return f"- {label}: {_format_value(old)} -> {_format_value(new)}"


def _seed_diff(current_manifest: Mapping[str, Any], prior_manifest: Mapping[str, Any]) -> list[str]:
    old = prior_manifest.get("seed")
    new = current_manifest.get("seed")
    if old == new:
        return []
    return [_diff_line("seed", old, new)]


def _cli_args_diff(
    current_manifest: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
) -> list[str]:
    current_cli = current_manifest.get("cli_args")
    prior_cli = prior_manifest.get("cli_args")
    if not isinstance(current_cli, Mapping) or not isinstance(prior_cli, Mapping):
        return []

    lines: list[str] = []
    for key in _CLI_DIFF_KEYS:
        old = prior_cli.get(key)
        new = current_cli.get(key)
        if old != new and (old is not None or new is not None):
            lines.append(_diff_line(f"cli_args.{key}", old, new))
    return lines


def _flatten_mapping(prefix: str, value: Mapping[str, Any], out: dict[str, Any]) -> None:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            _flatten_mapping(path, child, out)
        else:
            out[path] = child


def _wizard_diff(
    current_manifest: Mapping[str, Any],
    prior_manifest: Mapping[str, Any],
) -> list[str]:
    current_flat: dict[str, Any] = {}
    prior_flat: dict[str, Any] = {}

    for key in _WIZARD_KEYS:
        current_value = current_manifest.get(key)
        prior_value = prior_manifest.get(key)
        if isinstance(current_value, Mapping):
            _flatten_mapping(key, current_value, current_flat)
        if isinstance(prior_value, Mapping):
            _flatten_mapping(key, prior_value, prior_flat)

    lines: list[str] = []
    for path in sorted(set(current_flat) | set(prior_flat)):
        old = prior_flat.get(path)
        new = current_flat.get(path)
        if old != new:
            lines.append(_diff_line(path, old, new))
    return lines


def format_config_diff(
    current_manifest: Mapping[str, Any] | None,
    prior_manifest: Mapping[str, Any] | None,
) -> str:
    """Build one human-readable diff block across seed, CLI args, and wizard fields."""

    if not isinstance(current_manifest, Mapping) or not isinstance(prior_manifest, Mapping):
        return "No config differences available."

    lines = (
        _seed_diff(current_manifest, prior_manifest)
        + _cli_args_diff(current_manifest, prior_manifest)
        + _wizard_diff(current_manifest, prior_manifest)
    )
    if not lines:
        return "No config differences detected."
    return "\n".join(lines)


def build_metric_catalog(summary_df: pd.DataFrame) -> dict[str, float]:
    """Extract compact metrics used by compare/explain output."""

    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame")

    metrics: dict[str, float] = {}
    metric_columns = (
        SUMMARY_TE_COLUMN,
        SUMMARY_TRACKING_ERROR_LEGACY_COLUMN,
        SUMMARY_CVAR_COLUMN,
        SUMMARY_BREACH_PROB_COLUMN,
    )
    for column in metric_columns:
        if column in summary_df.columns and pd.api.types.is_numeric_dtype(summary_df[column]):
            metrics[column] = float(summary_df[column].mean())
    return metrics


def _heuristic_compare_text(
    *,
    config_diff: str,
    metrics_a: Mapping[str, float],
    metrics_b: Mapping[str, float],
    questions: str,
) -> str:
    deltas: list[str] = []
    for key in sorted(set(metrics_a) & set(metrics_b)):
        old = metrics_b[key]
        new = metrics_a[key]
        delta = new - old
        deltas.append(f"- {key}: prior={old:.4%}, current={new:.4%}, delta={delta:+.4%}")

    delta_block = "\n".join(deltas) if deltas else "- No shared numeric metrics found."
    return (
        "Run-to-run comparison summary:\n"
        f"Questions: {questions.strip() or 'default comparison questions'}\n\n"
        f"Config differences:\n{config_diff}\n\n"
        f"Metric deltas:\n{delta_block}"
    )


def _invoke_comparison_llm(
    prompt: str,
    *,
    provider_name: str | None,
    model_name: str | None,
    api_key: str | None,
) -> str:
    """Try an actual LLM invocation when langchain providers are installed."""

    # Optional dependency path; callers fall back to heuristic text on import/runtime errors.
    from pa_core.llm.provider import LLMProviderConfig, create_llm

    provider = (provider_name or "openai").strip().lower()
    if provider != "openai":
        # Provider-specific credentials are not yet threaded through this dashboard flow.
        provider = "openai"

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not isinstance(resolved_key, str) or not resolved_key.strip():
        raise ValueError("OPENAI_API_KEY is required for LLM comparison invocation.")

    llm = create_llm(
        LLMProviderConfig(
            provider_name=provider,
            credentials={"api_key": resolved_key},
            model_name=model_name or None,
        )
    )

    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_messages([("system", "{prompt}")]) | llm
    response = chain.invoke({"prompt": prompt})
    content = getattr(response, "content", None)
    return str(content or response).strip()


def compare_runs(
    *,
    current_summary: pd.DataFrame,
    current_manifest: Mapping[str, Any] | None,
    questions: str,
    provider_name: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None, CompareRunsPayload]:
    """Generate run comparison output and optional trace URL for Results panel."""

    prior_manifest, prior_manifest_path = load_prior_manifest(current_manifest)
    if prior_manifest is None:
        raise ValueError("No readable prior manifest found from manifest_data['previous_run'].")

    prior_summary, prior_summary_path = load_prior_summary(prior_manifest)
    if prior_summary is None:
        expected = str(prior_summary_path) if prior_summary_path else "<unset>"
        raise ValueError(f"No readable prior summary found at expected path: {expected}")

    config_diff = format_config_diff(current_manifest, prior_manifest)
    metrics_a = build_metric_catalog(current_summary)
    metrics_b = build_metric_catalog(prior_summary)

    prompt = build_comparison_prompt(
        item_a={"config_diff": config_diff, "metric_catalog": metrics_a, "label": "current"},
        item_b={"metric_catalog": metrics_b, "label": "prior"},
    )
    text = _heuristic_compare_text(
        config_diff=config_diff,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        questions=questions,
    )

    trace_url: str | None = None
    trace_id: str | None = None
    with langsmith_tracing_context(
        project_name="portable-alpha-results-compare",
        tags=("results", "comparison"),
        metadata={"provider": provider_name or "openai"},
    ):
        if os.getenv("LANGSMITH_API_KEY"):
            trace_id = uuid4().hex
        try:
            llm_text = _invoke_comparison_llm(
                prompt,
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key,
            )
            if llm_text:
                text = llm_text
        except Exception:
            # Keep dashboard responsive even without llm extras/credentials.
            pass

    if trace_id:
        trace_url = resolve_trace_url(trace_id)

    payload = CompareRunsPayload(
        config_diff=config_diff,
        metric_catalog_a=metrics_a,
        metric_catalog_b=metrics_b,
        prompt=prompt,
        questions=questions,
        prior_manifest_path=str(prior_manifest_path) if prior_manifest_path else None,
        prior_summary_path=str(prior_summary_path) if prior_summary_path else None,
    )
    return text, trace_url, payload


__all__ = [
    "CompareRunsPayload",
    "build_metric_catalog",
    "compare_runs",
    "format_config_diff",
    "load_prior_manifest",
    "load_prior_summary",
]
